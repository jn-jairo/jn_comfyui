from .base_model import get_model_loader

from .tts.models import (
    TtsSemanticModel,
    TtsSemanticTokenizer,
    TtsCoarseModel,
    TtsFineModel,
    TtsEncodecModel,
    TtsHubertModel,
    TtsHubertTokenizer,
)

from .vc.models import (
    VcFreeVCModel,
    VcSpeakerEncoderModel,
    VcWavLMModel,
)

MODELS_INFO = {
    "semantic_base": {
        "name": "semantic_base",
        "loader": TtsSemanticModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "text_2.pt",
        "filename": "semantic_base.pt",
        "dependencies": ["semantic_tokenizer"],
    },
    "coarse_base": {
        "name": "coarse_base",
        "loader": TtsCoarseModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "coarse_2.pt",
        "filename": "coarse_base.pt",
    },
    "fine_base": {
        "name": "fine_base",
        "loader": TtsFineModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "fine_2.pt",
        "filename": "fine_base.pt",
    },
    "semantic_small": {
        "name": "semantic_small",
        "loader": TtsSemanticModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "text.pt",
        "filename": "semantic_small.pt",
        "dependencies": ["semantic_tokenizer"],
    },
    "coarse_small": {
        "name": "coarse_small",
        "loader": TtsCoarseModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "coarse.pt",
        "filename": "coarse_small.pt",
    },
    "fine_small": {
        "name": "fine_small",
        "loader": TtsFineModel,
        "type": "hf",
        "repo_id": "suno/bark",
        "repo_path": "fine.pt",
        "filename": "fine_small.pt",
    },
    "semantic_tokenizer": {
        "name": "semantic_tokenizer",
        "loader": TtsSemanticTokenizer,
        "type": "transformers",
        "repo_id": "google-bert/bert-base-multilingual-cased",
    },
    "encodec": {
        "name": "encodec",
        "loader": TtsEncodecModel,
        "type": "transformers",
        "repo_id": "facebook/encodec_24khz",
    },
    "freevc": {
        "name": "freevc",
        "loader": VcFreeVCModel,
        "type": "hf",
        "repo_id": "jn-jairo/freevc",
        "repo_path": "freevc_24.pth",
        "filename": "freevc_24.pth",
        "dependencies": ["speaker_encoder"],
    },
    "speaker_encoder": {
        "name": "speaker_encoder",
        "loader": VcSpeakerEncoderModel,
        "type": "hf",
        "repo_id": "jn-jairo/freevc",
        "repo_path": "speaker_encoder.pt",
        "filename": "speaker_encoder.pt",
    },
    "wavlm_large": {
        "name": "wavlm_large",
        "loader": VcWavLMModel,
        "type": "hf",
        "repo_id": "jn-jairo/wavlm-large",
        "repo_path": "wavlm_large.pt",
        "filename": "wavlm_large.pt",
    },
    "hubert_base": {
        "name": "hubert_base",
        "loader": TtsHubertModel,
        "type": "url",
        "url": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "filename": "hubert_base.pt",
    },
    "hubert_tokenizer_en" : {
        "name": "hubert_tokenizer_en",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "GitMylo/bark-voice-cloning",
        "repo_path": "quantifier_V1_hubert_base_ls960_23.pth",
        "filename": "hubert_tokenizer_en.pth",
    },
    "hubert_tokenizer_de" : {
        "name": "hubert_tokenizer_de",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "CountFloyd/bark-voice-cloning-german-HuBERT-quantizer",
        "repo_path": "german-HuBERT-quantizer_14_epoch.pth",
        "filename": "hubert_tokenizer_de.pth",
    },
    "hubert_tokenizer_ja" : {
        "name": "hubert_tokenizer_ja",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "junwchina/bark-voice-cloning-japanese-HuBERT-quantizer",
        "repo_path": "japanese-HuBERT-quantizer_24_epoch.pth",
        "filename": "hubert_tokenizer_ja.pth",
    },
    "hubert_tokenizer_pl" : {
        "name": "hubert_tokenizer_pl",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "Hobis/bark-voice-cloning-polish-HuBERT-quantizer",
        "repo_path": "polish-HuBERT-quantizer_8_epoch.pth",
        "filename": "hubert_tokenizer_pl.pth",
    },
    "hubert_tokenizer_pt" : {
        "name": "hubert_tokenizer_pt",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "MadVoyager/bark-voice-cloning-portuguese-HuBERT-quantizer",
        "repo_path": "portuguese-HuBERT-quantizer_24_epoch.pth",
        "filename": "hubert_tokenizer_pt.pth",
    },
    "hubert_tokenizer_tr" : {
        "name": "hubert_tokenizer_tr",
        "loader": TtsHubertTokenizer,
        "type": "hf",
        "repo_id": "egeadam/bark-voice-cloning-turkish-HuBERT-quantizer",
        "repo_path": "turkish_model_epoch_14.pth",
        "filename": "hubert_tokenizer_tr.pth",
    },
}

def get_model(name, device=None, download=True, base_dir=None):
    info = MODELS_INFO[name]
    dependencies = {}
    if "dependencies" in info:
        for dependency in info["dependencies"]:
            dependencies[dependency] = MODELS_INFO[dependency]
    info["dependencies"] = dependencies

    return get_model_loader(info, device=device, download=download, base_dir=base_dir)
