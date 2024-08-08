import re
import numpy as np
import torch
import torch.nn.functional as F
import logging
import tqdm
import pytorch_seed

from transformers import BertTokenizer
from ...base_model import BaseModel
from .base import TtsBaseModel, inference_mode
from .gpt.gpt import GPT, GPTConfig
from ...utils import pbar_update

class TtsSemanticTokenizer(BaseModel):

    def load_model(self, *args, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(self.info["repo_id"], cache_dir=self.base_dir, local_files_only=not self.download)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

class TtsSemanticModel(TtsBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_class=GPT, config_class=GPTConfig, **kwargs)

    def _normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text):
        return self.dependencies["semantic_tokenizer"].encode(text, add_special_tokens=False)

    def _detokenize(self, enc_text):
        return self.dependencies["semantic_tokenizer"].decode(enc_text)

    def execute(self, text=[], context_semantic_tokens=None, seed=None, *args, **kwargs):
        if not isinstance(text, list):
            text = [text]

        semantic_tokens = []

        with pytorch_seed.SavedRNG(seed):
            for t in text:
                tokens = self._execute(text=t, context_semantic_tokens=context_semantic_tokens, *args, **kwargs)
                semantic_tokens.append(tokens)

                if context_semantic_tokens is None or isinstance(context_semantic_tokens, list) and len(context_semantic_tokens) == 0:
                    context_semantic_tokens = [tokens]

        return semantic_tokens

    def _execute(self, text, temperature=0.7, context_semantic_tokens=None, pbar_callback=None, *args, **kwargs):
        temp = temperature
        top_k = None
        top_p = None
        silent = False
        min_eos_p = 0.2
        max_gen_duration_s = None
        allow_early_stop = True
        use_kv_caching = True
        semantic_history = context_semantic_tokens[0] if isinstance(context_semantic_tokens, list) else context_semantic_tokens

        assert isinstance(text, str)

        text = self._normalize_whitespace(text)

        if semantic_history is not None:
            assert (
                isinstance(semantic_history, np.ndarray)
                and len(semantic_history.shape) == 1
                and len(semantic_history) > 0
                and semantic_history.min() >= 0
                and semantic_history.max() <= self.SEMANTIC_VOCAB_SIZE - 1
            )

        encoded_text = np.array(self._tokenize(text)) + self.TEXT_ENCODING_OFFSET

        device = next(self.model.parameters()).device

        if len(encoded_text) > 256:
            p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
            logging.warning(f"warning, text too long, lopping of last {p}%")
            encoded_text = encoded_text[:256]

        encoded_text = np.pad(
            encoded_text,
            (0, 256 - len(encoded_text)),
            constant_values=self.TEXT_PAD_TOKEN,
            mode="constant",
        )

        if semantic_history is not None:
            semantic_history = semantic_history.astype(np.int64)
            # lop off if history is too long, pad if needed
            semantic_history = semantic_history[-256:]
            semantic_history = np.pad(
                semantic_history,
                (0, 256 - len(semantic_history)),
                constant_values=self.SEMANTIC_PAD_TOKEN,
                mode="constant",
            )
        else:
            semantic_history = np.array([self.SEMANTIC_PAD_TOKEN] * 256)

        x = torch.from_numpy(
            np.hstack([
                encoded_text, semantic_history, np.array([self.SEMANTIC_INFER_TOKEN])
            ]).astype(np.int64)
        )[None]
        assert x.shape[1] == 256 + 256 + 1

        with inference_mode():
            x = x.to(device)

            n_tot_steps = 768
            # custom tqdm updates since we don't know when eos will occur
            pbar_update(pbar_callback, value=0, total=n_tot_steps)
            pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
            pbar_state = 0

            tot_generated_duration_s = 0
            kv_cache = None

            for n in range(n_tot_steps):
                if use_kv_caching and kv_cache is not None:
                    x_input = x[:, [-1]]
                else:
                    x_input = x

                logits, kv_cache = self.model(
                    x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
                )
                relevant_logits = logits[0, 0, :self.SEMANTIC_VOCAB_SIZE]

                if allow_early_stop:
                    relevant_logits = torch.hstack(
                        (relevant_logits, logits[0, 0, [self.SEMANTIC_PAD_TOKEN]]) # eos
                    )

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

                if allow_early_stop and (
                    item_next == self.SEMANTIC_VOCAB_SIZE
                    or (min_eos_p is not None and probs[-1] >= min_eos_p)
                ):
                    # eos found, so break
                    pbar.update(n - pbar_state)
                    pbar_update(pbar_callback, value=n+1, total=n_tot_steps)
                    break

                x = torch.cat((x, item_next[None]), dim=1)
                tot_generated_duration_s += 1 / self.SEMANTIC_RATE_HZ

                if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                    pbar.update(n - pbar_state)
                    pbar_update(pbar_callback, value=n+1, total=n_tot_steps)
                    break

                if n == n_tot_steps - 1:
                    pbar.update(n - pbar_state)
                    pbar_update(pbar_callback, value=n+1, total=n_tot_steps)
                    break

                del logits, relevant_logits, probs, item_next

                if n > pbar_state:
                    if n > pbar.total:
                        pbar.total = n
                    pbar.update(n - pbar_state)
                    pbar_update(pbar_callback, value=n+1, total=n_tot_steps)

                pbar_state = n

            pbar.total = n
            pbar.refresh()
            pbar.close()
            pbar_update(pbar_callback, value=n, total=n)

            out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]

            del x

        assert all(0 <= out) and all(out < self.SEMANTIC_VOCAB_SIZE)

        return out

