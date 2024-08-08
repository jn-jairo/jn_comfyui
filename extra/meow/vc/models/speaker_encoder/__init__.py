from ....base_model import BaseModel
from .voice_encoder import SpeakerEncoder

class VcSpeakerEncoderModel(BaseModel):

    def load_model(self, *args, **kwargs):
        self.model = SpeakerEncoder(self.get_file_path(), device=self.device)

