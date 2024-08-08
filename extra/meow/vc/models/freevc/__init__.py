import os
import torch
from torchaudio.functional import resample
import pytorch_seed

from ....base_model import BaseModel
from .models import SynthesizerTrn
from . import utils
from .mel_processing import mel_spectrogram_torch

class VcFreeVCModel(BaseModel):

    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000

    CONFIG_FILE = "freevc-24.json"

    def get_config(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(file_dir, "config", self.CONFIG_FILE)

        return utils.get_hparams_from_file(config_file)

    def load_model(self, *args, **kwargs):
        hps = self.get_config()

        model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).to(self.device)
        _ = model.eval()
        _ = utils.load_checkpoint(self.get_file_path(), model, None)

        self.model = model
        self.config = hps

        del model

    def model_to(self, device):
        if hasattr(self.model, "enc_spk") and self.model.enc_spk is not None:
            log = next(self.model.enc_spk.parameters()).device != device if hasattr(self.model.enc_spk, "parameters") else True
            if log:
                logging.info(f"Model '{self.name}' to " + repr(device))
            self.model.enc_spk.to(device)

    def encode_target(self, audio, *args, **kwargs):
        device = next(self.model.parameters()).device

        hps = self.config

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"].clone()

        (batches, channels, samples) = waveform.shape

        wf = waveform[0, :, :].clone().reshape(channels, samples)

        if wf.shape[0] > 1: # Stereo to mono if needed
            wf = wf.mean(0, keepdim=True)

        if sample_rate != self.INPUT_SAMPLE_RATE:
            wf = resample(wf, sample_rate, self.INPUT_SAMPLE_RATE)

        if hps.model.use_spk:
            tgt = self.dependencies["speaker_encoder"].model.embed_utterance(wf.squeeze(0).cpu().numpy())
            tgt = torch.from_numpy(tgt).unsqueeze(0)
        else:
            wf = wf.to(device)
            tgt = mel_spectrogram_torch(
                wf, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

        del waveform, wf
    
        return tgt.cpu().detach()

    def execute(self, target_tokens=None, source_tokens=[], seed=None, *args, **kwargs):
        if not isinstance(source_tokens, list):
            source_tokens = [source_tokens]

        device = next(self.model.parameters()).device

        hps = self.config

        tgt = target_tokens.clone().to(device)

        audios = []

        with pytorch_seed.SavedRNG(seed):
            for st in source_tokens:
                st = st.clone().to(device)

                if hps.model.use_spk:
                    audio = self.model.infer(st, g=tgt)
                else:
                    audio = self.model.infer(st, mel=tgt)
                audio = {"waveform": audio[0][0].data.cpu().float().detach().reshape(1, 1, -1), "sample_rate": self.OUTPUT_SAMPLE_RATE}

                audios.append(audio)

                del st

        del tgt

        return audios

