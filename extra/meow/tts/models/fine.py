import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pytorch_seed

from .base import TtsBaseModel, inference_mode
from .gpt.gpt_fine import FineGPT, FineGPTConfig
from ...utils import pbar_update

class TtsFineModel(TtsBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_class=FineGPT, config_class=FineGPTConfig, **kwargs)

    def execute(self, coarse_tokens=[], context_fine_tokens=None, seed=None, *args, **kwargs):
        if not isinstance(coarse_tokens, list):
            coarse_tokens = [coarse_tokens]

        fine_tokens = []

        with pytorch_seed.SavedRNG(seed):
            for ct in coarse_tokens:
                tokens = self._execute(coarse_tokens=ct, context_fine_tokens=context_fine_tokens, *args, **kwargs)
                fine_tokens.append(tokens)

                if context_fine_tokens is None or isinstance(context_fine_tokens, list) and len(context_fine_tokens) == 0:
                    context_fine_tokens = [tokens]

        return fine_tokens

    def _execute(self, coarse_tokens, temperature=0.7, context_fine_tokens=None, pbar_callback=None, *args, **kwargs):
        x_coarse_gen = coarse_tokens
        temp = temperature
        silent = False
        x_fine_history = context_fine_tokens[0] if isinstance(context_fine_tokens, list) else context_fine_tokens

        assert (
            isinstance(x_coarse_gen, np.ndarray)
            and len(x_coarse_gen.shape) == 2
            and 1 <= x_coarse_gen.shape[0] <= self.N_FINE_CODEBOOKS - 1
            and x_coarse_gen.shape[1] > 0
            and x_coarse_gen.min() >= 0
            and x_coarse_gen.max() <= self.CODEBOOK_SIZE - 1
        )

        if x_fine_history is not None:
            assert (
                isinstance(x_fine_history, np.ndarray)
                and len(x_fine_history.shape) == 2
                and x_fine_history.shape[0] == self.N_FINE_CODEBOOKS
                and x_fine_history.shape[1] >= 0
                and x_fine_history.min() >= 0
                and x_fine_history.max() <= self.CODEBOOK_SIZE - 1
            )

        n_coarse = x_coarse_gen.shape[0]

        device = next(self.model.parameters()).device

        # make input arr
        in_arr = np.vstack(
            [
                x_coarse_gen,
                np.zeros((self.N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
                + self.CODEBOOK_SIZE,  # padding
            ]
        ).astype(np.int32)

        # prepend history if available (max 512)
        if x_fine_history is not None:
            x_fine_history = x_fine_history.astype(np.int32)
            in_arr = np.hstack(
                [
                    x_fine_history[:, -512:].astype(np.int32),
                    in_arr,
                ]
            )
            n_history = x_fine_history[:, -512:].shape[1]
        else:
            n_history = 0

        n_remove_from_end = 0

        # need to pad if too short (since non-causal model)
        if in_arr.shape[1] < 1024:
            n_remove_from_end = 1024 - in_arr.shape[1]
            in_arr = np.hstack(
                [
                    in_arr,
                    np.zeros((self.N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + self.CODEBOOK_SIZE,
                ]
            )

        # we can be lazy about fractional loop and just keep overwriting codebooks
        n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1

        with inference_mode():
            in_arr = torch.tensor(in_arr.T).to(device)

            for n in tqdm.tqdm(range(n_loops), disable=silent):
                start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
                start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
                rel_start_fill_idx = start_fill_idx - start_idx
                in_buffer = in_arr[start_idx : start_idx + 1024, :][None]

                for nn in range(n_coarse, self.N_FINE_CODEBOOKS):
                    logits = self.model(nn, in_buffer)

                    if temp is None:
                        relevant_logits = logits[0, rel_start_fill_idx:, :self.CODEBOOK_SIZE]
                        codebook_preds = torch.argmax(relevant_logits, -1)
                    else:
                        relevant_logits = logits[0, :, :self.CODEBOOK_SIZE] / temp
                        probs = F.softmax(relevant_logits, dim=-1)
                        codebook_preds = torch.multinomial(
                            probs[rel_start_fill_idx:1024], num_samples=1
                        ).reshape(-1)

                    codebook_preds = codebook_preds.to(torch.int32)
                    in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds

                    del logits, codebook_preds

                # transfer over info into model_in and convert to numpy
                for nn in range(n_coarse, self.N_FINE_CODEBOOKS):
                    in_arr[
                        start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
                    ] = in_buffer[0, rel_start_fill_idx:, nn]

                del in_buffer
                pbar_update(pbar_callback, value=int(n+1), total=int(n_loops))

            pbar_update(pbar_callback, value=int(n_loops), total=int(n_loops))

            gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
            del in_arr

        gen_fine_arr = gen_fine_arr[:, n_history:]

        if n_remove_from_end > 0:
            gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]

        assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]

        return gen_fine_arr

