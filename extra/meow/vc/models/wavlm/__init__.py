import torch
from torchaudio.functional import resample

from ....base_model import BaseModel
from .WavLM import WavLM, WavLMConfig
from ..freevc import VcFreeVCModel

class VcWavLMModel(BaseModel):

    def load_model(self, *args, **kwargs):
        checkpoint = torch.load(self.get_file_path())

        cfg = WavLMConfig(checkpoint['cfg'])

        model = WavLM(cfg).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        self.model = model

        del model

    def execute(self, audio=[], *args, **kwargs):
        if not isinstance(audio, list):
            audio = [audio]

        tokens = []

        for a in audio:
            for t in self._execute(audio=a, *args, **kwargs):
                tokens.append(t)

        return tokens

    def _execute(self, audio, *args, **kwargs):
        device = next(self.model.parameters()).device

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"].clone().to(device)

        (batches, channels, samples) = waveform.shape

        tokens = []

        for b in range(batches):
            wf = waveform[b, :, :].clone().reshape(channels, samples)

            if wf.shape[0] > 1: # Stereo to mono if needed
                wf = wf.mean(0, keepdim=True)

            if sample_rate != VcFreeVCModel.INPUT_SAMPLE_RATE:
                wf = resample(wf, sample_rate, VcFreeVCModel.INPUT_SAMPLE_RATE)

            with torch.no_grad():
                c = self.model.extract_features(wf)[0]
            c = c.transpose(1, 2)

            tokens.append(c.cpu().detach())

            del wf, c

        del waveform

        return tokens

