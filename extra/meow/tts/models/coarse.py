import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pytorch_seed

from .base import TtsBaseModel, inference_mode
from .gpt.gpt import GPT, GPTConfig
from ...utils import pbar_update

class TtsCoarseModel(TtsBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_class=GPT, config_class=GPTConfig, **kwargs)

    def _flatten_codebooks(self, arr, offset_size=None):
        if offset_size is None:
            offset_size = self.CODEBOOK_SIZE
        assert len(arr.shape) == 2
        arr = arr.copy()
        if offset_size is not None:
            for n in range(1, arr.shape[0]):
                arr[n, :] += offset_size * n
        flat_arr = arr.ravel("F")
        return flat_arr

    def execute(self, semantic_tokens=[], context_coarse_tokens=None, seed=None, *args, **kwargs):
        if not isinstance(semantic_tokens, list):
            semantic_tokens = [semantic_tokens]

        coarse_tokens = []

        with pytorch_seed.SavedRNG(seed):
            for st in semantic_tokens:
                tokens = self._execute(semantic_tokens=st, context_coarse_tokens=context_coarse_tokens, *args, **kwargs)
                coarse_tokens.append(tokens)

                if context_coarse_tokens is None or isinstance(context_coarse_tokens, list) and len(context_coarse_tokens) == 0:
                    context_coarse_tokens = [tokens]

        return coarse_tokens

    def _execute(self, semantic_tokens, temperature=0.7, context_semantic_tokens=None, context_coarse_tokens=None, pbar_callback=None, *args, **kwargs):
        x_semantic = semantic_tokens
        temp = temperature
        top_k = None
        top_p = None
        silent = False
        max_coarse_history = 630  # min 60 (faster) max 630 (more context)
        sliding_window_len = 60
        use_kv_caching = True
        x_semantic_history = context_semantic_tokens[0] if isinstance(context_semantic_tokens, list) else context_semantic_tokens
        x_coarse_history = context_coarse_tokens[0] if isinstance(context_coarse_tokens, list) else context_coarse_tokens

        assert (
            isinstance(x_semantic, np.ndarray)
            and len(x_semantic.shape) == 1
            and len(x_semantic) > 0
            and x_semantic.min() >= 0
            and x_semantic.max() <= self.SEMANTIC_VOCAB_SIZE - 1
        )
        assert 60 <= max_coarse_history <= 630
        assert max_coarse_history + sliding_window_len <= 1024 - 256

        semantic_to_coarse_ratio = self.COARSE_RATE_HZ / self.SEMANTIC_RATE_HZ * self.N_COARSE_CODEBOOKS
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))

        if x_semantic_history is not None and x_coarse_history is not None:
            assert (
                isinstance(x_semantic_history, np.ndarray)
                and len(x_semantic_history.shape) == 1
                and len(x_semantic_history) > 0
                and x_semantic_history.min() >= 0
                and x_semantic_history.max() <= self.SEMANTIC_VOCAB_SIZE - 1
                and isinstance(x_coarse_history, np.ndarray)
                and len(x_coarse_history.shape) == 2
                and x_coarse_history.shape[0] == self.N_COARSE_CODEBOOKS
                and x_coarse_history.shape[-1] >= 0
                and x_coarse_history.min() >= 0
                and x_coarse_history.max() <= self.CODEBOOK_SIZE - 1
                and (
                    round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                    == round(semantic_to_coarse_ratio / self.N_COARSE_CODEBOOKS, 1)
                )
            )
            x_coarse_history = self._flatten_codebooks(x_coarse_history) + self.SEMANTIC_VOCAB_SIZE
            # trim histories correctly
            n_semantic_hist_provided = np.min(
                [
                    max_semantic_history,
                    len(x_semantic_history) - len(x_semantic_history) % 2,
                    int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
                ]
            )
            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
            x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
            x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
            # TODO: bit of a hack for time alignment (sounds better)
            x_coarse_history = x_coarse_history[:-2]
        else:
            x_semantic_history = np.array([], dtype=np.int32)
            x_coarse_history = np.array([], dtype=np.int32)

        device = next(self.model.parameters()).device

        # start loop
        n_steps = int(
            round(
                np.floor(len(x_semantic) * semantic_to_coarse_ratio / self.N_COARSE_CODEBOOKS)
                * self.N_COARSE_CODEBOOKS
            )
        )
        assert n_steps > 0 and n_steps % self.N_COARSE_CODEBOOKS == 0

        x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
        x_coarse = x_coarse_history.astype(np.int32)

        base_semantic_idx = len(x_semantic_history)

        with inference_mode():
            x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
            x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)

            n_window_steps = int(np.ceil(n_steps / sliding_window_len))
            n_step = 0

            for n in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
                semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))

                # pad from right side
                x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
                x_in = x_in[:, :256]
                x_in = F.pad(
                    x_in,
                    (0, 256 - x_in.shape[-1]),
                    "constant",
                    self.COARSE_SEMANTIC_PAD_TOKEN,
                )
                x_in = torch.hstack(
                    [
                        x_in,
                        torch.tensor([self.COARSE_INFER_TOKEN])[None].to(device),
                        x_coarse_in[:, -max_coarse_history:],
                    ]
                )

                kv_cache = None

                for _ in range(sliding_window_len):
                    if n_step >= n_steps:
                        continue

                    is_major_step = n_step % self.N_COARSE_CODEBOOKS == 0

                    if use_kv_caching and kv_cache is not None:
                        x_input = x_in[:, [-1]]
                    else:
                        x_input = x_in

                    logits, kv_cache = self.model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                    logit_start_idx = (
                        self.SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * self.CODEBOOK_SIZE
                    )
                    logit_end_idx = (
                        self.SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * self.CODEBOOK_SIZE
                    )
                    relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]

                    if top_p is not None:
                        # faster to convert to numpy
                        original_device = relevant_logits.device
                        relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                        sorted_indices = np.argsort(relevant_logits)[::-1]
                        sorted_logits = relevant_logits[sorted_indices]
                        cumulative_probs = np.cumsum(softmax(sorted_logits))
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                        sorted_indices_to_remove[0] = False
                        relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                        relevant_logits = torch.from_numpy(relevant_logits)
                        relevant_logits = relevant_logits.to(original_device)

                    if top_k is not None:
                        v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                        relevant_logits[relevant_logits < v[-1]] = -float("Inf")

                    probs = F.softmax(relevant_logits / temp, dim=-1)

                    item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                    item_next += logit_start_idx

                    x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                    x_in = torch.cat((x_in, item_next[None]), dim=1)

                    del logits, relevant_logits, probs, item_next
                    n_step += 1

                del x_in
                pbar_update(pbar_callback, value=n+1, total=n_window_steps)

            del x_semantic_in
            pbar_update(pbar_callback, value=n_window_steps, total=n_window_steps)

            gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history) :]
            del x_coarse_in

        assert len(gen_coarse_arr) == n_steps

        gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, self.N_COARSE_CODEBOOKS).T - self.SEMANTIC_VOCAB_SIZE

        for n in range(1, self.N_COARSE_CODEBOOKS):
            gen_coarse_audio_arr[n, :] -= n * self.CODEBOOK_SIZE

        return gen_coarse_audio_arr

