import torch
import logging

from ....base_model import BaseModel
from .hubert import CustomHubert
from .hubert_tokenizer import CustomTokenizer

class TtsHubertTokenizer(BaseModel):

    def load_model(self, *args, **kwargs):
        self.model = CustomTokenizer.load_from_checkpoint(self.get_file_path(), map_location=self.device)

    def execute(self, semantic_vectors=[], *args, **kwargs):
        if not isinstance(semantic_vectors, list):
            semantic_vectors = [semantic_vectors]

        semantic_tokens = []

        for sv in semantic_vectors:
            semantic_tokens.append(self._execute(semantic_vectors=sv, *args, **kwargs))

        return semantic_tokens

    def _execute(self, semantic_vectors, *args, **kwargs):
        device = next(self.model.parameters()).device

        sv = torch.from_numpy(semantic_vectors)
        sv = sv.to(device)

        semantic_tokens = self.model.get_token(sv).cpu().detach().numpy()

        del sv

        return semantic_tokens

class TtsHubertModel(BaseModel):

    def load_model(self, *args, **kwargs):
        self.model = CustomHubert(self.get_file_path(), device=self.device)

    def model_to(self, device):
        log = next(self.model.model.parameters()).device != device if hasattr(self.model.model, "parameters") else True
        if log:
            logging.info(f"Model '{self.name}' to " + repr(device))
        self.model.model.to(device)

    def execute(self, audio=[], *args, **kwargs):
        if not isinstance(audio, list):
            audio = [audio]

        semantic_vectors = []

        for a in audio:
            for sv in self._execute(audio=a, *args, **kwargs):
                semantic_vectors.append(sv)

        return semantic_vectors

    def _execute(self, audio, *args, **kwargs):
        device = next(self.model.model.parameters()).device

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"].clone().to(device)

        (batches, channels, samples) = waveform.shape

        semantic_vectors = []

        for b in range(batches):
            wf = waveform[b, :, :].clone().reshape(channels, samples)

            if wf.shape[0] > 1: # Stereo to mono if needed
                wf = wf.mean(0, keepdim=True)

            sv = self.model.forward(wf, input_sample_hz=sample_rate)

            semantic_vectors.append(sv.cpu().detach().numpy())

        del waveform

        return semantic_vectors

