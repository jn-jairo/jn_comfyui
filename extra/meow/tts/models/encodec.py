import torch
import torchaudio
import warnings

from ...base_model import BaseModel
from .base import TtsBaseModel
from transformers import EncodecModel

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

class TtsEncodecModel(BaseModel):

    def load_model(self, *args, **kwargs):
        self.model = EncodecModel.from_pretrained(self.info["repo_id"], cache_dir=self.base_dir, local_files_only=not self.download)

    def encode(self, audio=[], *args, **kwargs):
        if not isinstance(audio, list):
            audio = [audio]

        coarse_tokens = []
        fine_tokens = []

        for a in audio:
            cts, fts = self._encode(audio=a, *args, **kwargs)
            for ct in cts:
                coarse_tokens.append(ct)
            for ft in fts:
                fine_tokens.append(ft)

        return (coarse_tokens, fine_tokens)

    def _encode(self, audio, *args, **kwargs):
        device = next(self.model.parameters()).device

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"].clone().to(device)

        (batches, channels, samples) = waveform.shape

        coarse_tokens = []
        fine_tokens = []

        for b in range(batches):
            wf = waveform[b, :, :].clone().reshape(channels, samples)

            wf = self.convert_audio(wf, sample_rate, self.model.config.sampling_rate, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                encoded_frames = self.model.encode(wf, bandwidth=6.0).audio_codes

            del wf

            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

            fine_tokens.append(codes.cpu().detach().numpy())
            coarse_tokens.append(codes[:2, :].cpu().detach().numpy())

        del waveform

        return (coarse_tokens, fine_tokens)

    def decode(self, fine_tokens=[], *args, **kwargs):
        if not isinstance(fine_tokens, list):
            fine_tokens = [fine_tokens]

        audios = []

        for ft in fine_tokens:
            audios.append(self._decode(fine_tokens=ft, *args, **kwargs))

        return audios

    def _decode(self, fine_tokens, *args, **kwargs):
        device = next(self.model.parameters()).device

        arr = torch.from_numpy(fine_tokens)[None]
        arr = arr.to(device)
        arr = arr.transpose(0, 1)

        out = self.model.decoder(self.model.quantizer.decode(arr))

        del arr

        return {"waveform": out.cpu().detach(), "sample_rate": TtsBaseModel.SAMPLE_RATE}

    def convert_audio(self, wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
        assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
        assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
        *shape, channels, length = wav.shape
        if target_channels == 1:
            wav = wav.mean(-2, keepdim=True)
        elif target_channels == 2:
            wav = wav.expand(*shape, target_channels, length)
        elif channels == 1:
            wav = wav.expand(target_channels, -1)
        else:
            raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav.cpu())
        return wav

