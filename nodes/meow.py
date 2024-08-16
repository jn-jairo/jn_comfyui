import os
import folder_paths
from functools import reduce
import comfy
import glob
import numpy as np
import hashlib
import torch

from ..utils import (
    CATEGORY_AUDIO_MEOW,
    CATEGORY_AUDIO_MEOW_TTS,
    CATEGORY_AUDIO_MEOW_VC,
    CATEGORY_AUDIO_MEOW_HRTF,
    DEVICES,
    get_device,
)

from ..extra.meow.model import get_model, HRTF_CIPIC_MODELS
from ..extra.meow.base_model import ModelDeviceContext
from ..extra.meow.utils import (
    sentence_split,
    LANGUAGES_NAMES,
    TTS_LANGUAGES,
    TTS_VC_LANGUAGES,
    NLTK_LANGUAGES,
)

from .audio import batch_to_array

MODELS = [
    "base",
    "small",
]

def get_base_dir():
    return folder_paths.get_folder_paths("meow")[0]

class JN_MeowTtsModel:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_MODEL", "MEOW_TTS_COARSE_MODEL", "MEOW_TTS_FINE_MODEL")
    RETURN_NAMES = ("semantic", "coarse", "fine")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODELS,),
                "device": (DEVICES,),
            },
        }

    def run(self, model, device):
        device = get_device(device)

        base_dir = get_base_dir()

        semantic_model = get_model("semantic_" + model, device=device, base_dir=base_dir)
        coarse_model = get_model("coarse_" + model, device=device, base_dir=base_dir)
        fine_model = get_model("fine_" + model, device=device, base_dir=base_dir)

        semantic_model.load()
        coarse_model.load()
        fine_model.load()

        return (semantic_model, coarse_model, fine_model)

class JN_MeowTtsModelSemantic:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_MODEL",)
    RETURN_NAMES = ("semantic",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODELS,),
                "device": (DEVICES,),
            },
        }

    def run(self, model, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("semantic_" + model, device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTtsModelCoarse:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_COARSE_MODEL",)
    RETURN_NAMES = ("coarse",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODELS,),
                "device": (DEVICES,),
            },
        }

    def run(self, model, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("coarse_" + model, device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTtsModelFine:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_FINE_MODEL",)
    RETURN_NAMES = ("fine",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODELS,),
                "device": (DEVICES,),
            },
        }

    def run(self, model, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("fine_" + model, device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTtsModelEncodec:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_ENCODEC_MODEL",)
    RETURN_NAMES = ("encodec",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (DEVICES,),
            },
        }

    def run(self, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("encodec", device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTtsModelHubert:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_HUBERT_MODEL",)
    RETURN_NAMES = ("hubert",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (DEVICES,),
            },
        }

    def run(self, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("hubert_base", device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTtsTokenizerHubert:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_HUBERT_TOKENIZER",)
    RETURN_NAMES = ("hubert_tokenizer",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        languages = [f"{key}: {LANGUAGES_NAMES[key]}" for key in TTS_VC_LANGUAGES]
        return {
            "required": {
                "device": (DEVICES,),
                "language": (languages,)
            },
        }

    def run(self, device, language):
        device = get_device(device)

        base_dir = get_base_dir()

        language = language.split(":")[0]

        model = get_model(f"hubert_tokenizer_{language}", device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowVcModelWavLM:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_WAVLM_MODEL",)
    RETURN_NAMES = ("wavlm",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (DEVICES,),
            },
        }

    def run(self, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("wavlm_large", device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowVcModelFreeVC:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_FREEVC_MODEL",)
    RETURN_NAMES = ("freevc",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (DEVICES,),
            },
        }

    def run(self, device):
        device = get_device(device)

        base_dir = get_base_dir()

        model = get_model("freevc", device=device, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowHrtfModel:
    CATEGORY = CATEGORY_AUDIO_MEOW_HRTF
    RETURN_TYPES = ("MEOW_HRTF_MODEL",)
    RETURN_NAMES = ("hrtf",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        models = [f"hrtf_cipic_{i}" for i in HRTF_CIPIC_MODELS]

        return {
            "required": {
                "model": (models,),
            },
        }

    def run(self, model):
        base_dir = get_base_dir()

        model = get_model(model, base_dir=base_dir)
        model.load()

        return (model,)

class JN_MeowTts:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_TOKENS", "MEOW_TTS_COARSE_TOKENS", "MEOW_TTS_FINE_TOKENS", "ARRAY")
    RETURN_NAMES = ("semantic_tokens", "coarse_tokens", "fine_tokens", "audios")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "semantic": ("MEOW_TTS_SEMANTIC_MODEL",),
                "coarse": ("MEOW_TTS_COARSE_MODEL",),
                "fine": ("MEOW_TTS_FINE_MODEL",),
                "encodec": ("MEOW_TTS_ENCODEC_MODEL",),
                "texts": ("*", {"multiple": True}),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "temperature_semantic": ("FLOAT", {"default": 0.7, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
                "temperature_coarse": ("FLOAT", {"default": 0.7, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
                "temperature_fine": ("FLOAT", {"default": 0.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
            },
            "optional" :{
                "context_semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
                "context_coarse_tokens": ("MEOW_TTS_COARSE_TOKENS",),
                "context_fine_tokens": ("MEOW_TTS_FINE_TOKENS",),
            },
        }

    def run(self, semantic, coarse, fine, encodec, texts, device, seed,
            temperature_semantic=0.7, temperature_coarse=0.7, temperature_fine=0.5,
            context_semantic_tokens=None, context_coarse_tokens=None, context_fine_tokens=None,
            **kwargs):
        texts = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), texts, [None])
        texts = [text for text in texts if text is not None]

        semantic_tokens = JN_MeowTtsSemantic().run(
            semantic=semantic,
            device=device,
            texts=texts,
            seed=seed,
            temperature=temperature_semantic,
            context_semantic_tokens=context_semantic_tokens,
        )[0]

        if context_semantic_tokens is None or isinstance(context_semantic_tokens, list) and len(context_semantic_tokens) == 0:
            context_semantic_tokens = semantic_tokens

        coarse_tokens = JN_MeowTtsCoarse().run(
            coarse=coarse,
            semantic_tokens=semantic_tokens,
            device=device,
            seed=seed,
            temperature=temperature_coarse,
            context_semantic_tokens=context_semantic_tokens,
            context_coarse_tokens=context_coarse_tokens,
        )[0]

        fine_tokens = JN_MeowTtsFine().run(
            fine=fine,
            coarse_tokens=coarse_tokens,
            device=device,
            seed=seed,
            temperature=temperature_fine,
            context_fine_tokens=context_fine_tokens,
        )[0]

        audio = JN_MeowTtsDecode().run(
            encodec=encodec,
            fine_tokens=fine_tokens,
            device=device,
        )[0]

        return (semantic_tokens, coarse_tokens, fine_tokens, audio)

class JN_MeowTtsSemantic:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_TOKENS",)
    RETURN_NAMES = ("semantic_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "semantic": ("MEOW_TTS_SEMANTIC_MODEL",),
                "texts": ("*", {"multiple": True}),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
            },
            "optional" :{
                "context_semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
            },
        }

    def run(self, semantic, texts, device, seed, temperature=0.7, context_semantic_tokens=None):
        texts = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), texts, [None])
        texts = [text for text in texts if text is not None]

        device = get_device(device)

        pbar = comfy.utils.ProgressBar(0)
        def pbar_callback(value, total):
            pbar.update_absolute(value=value, total=total)

        with ModelDeviceContext(semantic, device):
            semantic_tokens = semantic.execute(
                text=texts,
                seed=seed,
                temperature=temperature,
                context_semantic_tokens=context_semantic_tokens,
                pbar_callback=pbar_callback,
            )

        comfy.model_management.soft_empty_cache(True)

        return (semantic_tokens,)

class JN_MeowTtsCoarse:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_COARSE_TOKENS",)
    RETURN_NAMES = ("coarse_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coarse": ("MEOW_TTS_COARSE_MODEL",),
                "semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
            },
            "optional" :{
                "context_semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
                "context_coarse_tokens": ("MEOW_TTS_COARSE_TOKENS",),
            },
        }

    def run(self, coarse, semantic_tokens, device, seed, temperature=0.7, context_semantic_tokens=None, context_coarse_tokens=None):
        device = get_device(device)

        pbar = comfy.utils.ProgressBar(0)
        def pbar_callback(value, total):
            pbar.update_absolute(value=value, total=total)

        with ModelDeviceContext(coarse, device):
            coarse_tokens = coarse.execute(
                semantic_tokens=semantic_tokens,
                seed=seed,
                temperature=temperature,
                context_semantic_tokens=context_semantic_tokens,
                context_coarse_tokens=context_coarse_tokens,
                pbar_callback=pbar_callback,
            )

        comfy.model_management.soft_empty_cache(True)

        return (coarse_tokens,)

class JN_MeowTtsFine:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_FINE_TOKENS",)
    RETURN_NAMES = ("fine_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fine": ("MEOW_TTS_FINE_MODEL",),
                "coarse_tokens": ("MEOW_TTS_COARSE_TOKENS",),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
            },
            "optional" :{
                "context_fine_tokens": ("MEOW_TTS_FINE_TOKENS",),
            },
        }

    def run(self, fine, coarse_tokens, device, seed, temperature=0.5, context_fine_tokens=None):
        device = get_device(device)

        pbar = comfy.utils.ProgressBar(0)
        def pbar_callback(value, total):
            pbar.update_absolute(value=value, total=total)

        with ModelDeviceContext(fine, device):
            fine_tokens = fine.execute(
                coarse_tokens=coarse_tokens,
                seed=seed,
                temperature=temperature,
                context_fine_tokens=context_fine_tokens,
                pbar_callback=pbar_callback,
            )

        comfy.model_management.soft_empty_cache(True)

        return (fine_tokens,)

class JN_MeowTtsDecode:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "encodec": ("MEOW_TTS_ENCODEC_MODEL",),
                "fine_tokens": ("MEOW_TTS_FINE_TOKENS",),
                "device": (DEVICES,),
            },
        }

    def run(self, encodec, fine_tokens, device):
        device = get_device(device)

        with ModelDeviceContext(encodec, device):
            audio = encodec.decode(fine_tokens=fine_tokens)

        comfy.model_management.soft_empty_cache(True)

        return (audio,)

class JN_MeowSentenceSplit:
    CATEGORY = CATEGORY_AUDIO_MEOW
    RETURN_TYPES = ("ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        languages = [f"{key}: {LANGUAGES_NAMES[key]}" for key in NLTK_LANGUAGES.keys()]
        return {
            "required": {
                "text": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
                "language": (languages,),
                "max_characters": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "extra_separators": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
            },
        }

    def run(self, text, language, max_characters=0, extra_separators=""):
        extra_separators = [separator for separator in extra_separators.split("\n") if separator.lstrip().rstrip() != ""]

        sentences = sentence_split(text, language=language.split(":")[0], max_characters=max_characters, extra_separators=extra_separators)

        return (sentences,)

class JN_MeowTtsSaveContext:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    FORMATS = [
        "meow",
        "bark",
    ]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
                "coarse_tokens": ("MEOW_TTS_COARSE_TOKENS",),
                "fine_tokens": ("MEOW_TTS_FINE_TOKENS",),
                "filename_prefix": ("STRING", {"default": "meow_tts/ComfyUI"}),
                "format": (s.FORMATS,),
            },
        }

    def run(self, semantic_tokens, coarse_tokens, fine_tokens, filename_prefix, format="meow"):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        semantic_tokens = [semantic_tokens] if not isinstance(semantic_tokens, list) else semantic_tokens
        coarse_tokens = [coarse_tokens] if not isinstance(coarse_tokens, list) else coarse_tokens
        fine_tokens = [fine_tokens] if not isinstance(fine_tokens, list) else fine_tokens

        for i in range(len(semantic_tokens)):
            file = filename.replace("%batch_num%", str(i))
            file = f"{file}_{counter:05}_.npz"
            file = os.path.join(full_output_folder, file)

            if format == "bark":
                context = {
                    "semantic_prompt": semantic_tokens[i],
                    "coarse_prompt": coarse_tokens[i],
                    "fine_prompt": fine_tokens[i],
                }
            else:
                context = {
                    "semantic": semantic_tokens[i],
                    "coarse": coarse_tokens[i],
                    "fine": fine_tokens[i],
                }

            np.savez(file, **context)

            counter += 1

        return {"ui": {}}

class JN_MeowTtsLoadContext:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_TOKENS", "MEOW_TTS_COARSE_TOKENS", "MEOW_TTS_FINE_TOKENS")
    RETURN_NAMES = ("semantic_tokens", "coarse_tokens", "fine_tokens")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = sorted(list(set([d for d in glob.iglob("**/**", root_dir=input_dir, recursive=True) if os.path.isfile(os.path.join(input_dir, d)) and d.endswith(".npz")])))
        return {
            "required": {
                "context": (files,),
            },
        }

    def run(self, context):
        semantic_tokens = None
        coarse_tokens = None
        fine_tokens = None

        file_path = folder_paths.get_annotated_filepath(context)
        context = np.load(file_path)

        if context:
            if "semantic" in context:
                semantic_tokens = context["semantic"]
            elif "semantic_prompt" in context:
                semantic_tokens = context["semantic_prompt"]

            if "coarse" in context:
                coarse_tokens = context["coarse"]
            elif "coarse_prompt" in context:
                coarse_tokens = context["coarse_prompt"]

            if "fine" in context:
                fine_tokens = context["fine"]
            elif "fine_prompt" in context:
                fine_tokens = context["fine_prompt"]

        return (semantic_tokens, coarse_tokens, fine_tokens)

    @classmethod
    def IS_CHANGED(s, context):
        file_path = folder_paths.get_annotated_filepath(context)
        m = hashlib.sha256()
        with open(file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, context):
        if not folder_paths.exists_annotated_filepath(context):
            return "Invalid context file: {}".format(context)

        return True

class JN_MeowTtsAudioToContext:
    CATEGORY = CATEGORY_AUDIO_MEOW_TTS
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_TOKENS", "MEOW_TTS_COARSE_TOKENS", "MEOW_TTS_FINE_TOKENS")
    RETURN_NAMES = ("semantic_tokens", "coarse_tokens", "fine_tokens")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hubert": ("MEOW_TTS_HUBERT_MODEL",),
                "hubert_tokenizer": ("MEOW_TTS_HUBERT_TOKENIZER",),
                "encodec": ("MEOW_TTS_ENCODEC_MODEL",),
                "audios": ("*", {"multiple": True}),
                "device": (DEVICES,),
            },
        }

    def run(self, hubert, hubert_tokenizer, encodec, audios, device):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        semantic_tokens = None
        coarse_tokens = None
        fine_tokens = None

        device = get_device(device)

        with ModelDeviceContext(hubert, device):
            semantic_vectors = hubert.execute(audio=audios)

        comfy.model_management.soft_empty_cache(True)

        with ModelDeviceContext(hubert_tokenizer, device):
            semantic_tokens = hubert_tokenizer.execute(semantic_vectors=semantic_vectors)

        comfy.model_management.soft_empty_cache(True)

        with ModelDeviceContext(encodec, device):
            coarse_tokens, fine_tokens = encodec.encode(audio=audios)

        comfy.model_management.soft_empty_cache(True)

        return (semantic_tokens, coarse_tokens, fine_tokens)

class JN_MeowVc:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_SOURCE_TOKENS", "MEOW_VC_TARGET_TOKENS", "ARRAY",)
    RETURN_NAMES = ("source_tokens", "target_tokens", "audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wavlm": ("MEOW_VC_WAVLM_MODEL",),
                "freevc": ("MEOW_VC_FREEVC_MODEL",),
                "target_audio": ("*",),
                "audios": ("*", {"multiple": True}),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    def run(self, wavlm, freevc, target_audio, audios, device, seed):
        source_tokens = JN_MeowVcEncodeSource().run(
            wavlm=wavlm,
            audios=audios,
            device=device,
        )[0]

        comfy.model_management.soft_empty_cache(True)

        target_tokens = JN_MeowVcEncodeTarget().run(
            freevc=freevc,
            audio=target_audio,
            device=device,
        )[0]

        comfy.model_management.soft_empty_cache(True)

        output_audios = JN_MeowVcConvertVoice().run(
            freevc=freevc,
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            device=device,
            seed=seed,
        )[0]

        comfy.model_management.soft_empty_cache(True)

        return (source_tokens, target_tokens, output_audios,)

class JN_MeowVcEncodeSource:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_SOURCE_TOKENS",)
    RETURN_NAMES = ("source_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wavlm": ("MEOW_VC_WAVLM_MODEL",),
                "audios": ("*", {"multiple": True}),
                "device": (DEVICES,),
            },
        }

    def run(self, wavlm, audios, device):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        device = get_device(device)

        with ModelDeviceContext(wavlm, device):
            source_tokens = wavlm.execute(audio=audios)

        comfy.model_management.soft_empty_cache(True)

        return (source_tokens,)

class JN_MeowVcEncodeTarget:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_TARGET_TOKENS",)
    RETURN_NAMES = ("target_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "freevc": ("MEOW_VC_FREEVC_MODEL",),
                "audio": ("*",),
                "device": (DEVICES,),
            },
        }

    def run(self, freevc, audio, device):
        audios = [audio]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])
        audio = audios[0]

        device = get_device(device)

        with ModelDeviceContext(freevc, device):
            target_tokens = freevc.encode_target(audio=audio)

        comfy.model_management.soft_empty_cache(True)

        return (target_tokens,)

class JN_MeowVcConvertVoice:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "freevc": ("MEOW_VC_FREEVC_MODEL",),
                "target_tokens": ("MEOW_VC_TARGET_TOKENS",),
                "source_tokens": ("MEOW_VC_SOURCE_TOKENS",),
                "device": (DEVICES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    def run(self, freevc, target_tokens, source_tokens, device, seed):
        device = get_device(device)

        with ModelDeviceContext(freevc, device):
            audios = freevc.execute(target_tokens=target_tokens, source_tokens=source_tokens, seed=seed)

        comfy.model_management.soft_empty_cache(True)

        return (audios,)

class JN_MeowVcSaveSpeaker:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_tokens": ("MEOW_VC_TARGET_TOKENS",),
                "filename_prefix": ("STRING", {"default": "meow_vc/ComfyUI"}),
            },
        }

    def run(self, target_tokens, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.npz"
        file = os.path.join(full_output_folder, file)

        speaker = {
            "speaker": target_tokens[i],
        }

        np.savez(file, **speaker)

        return {"ui": {}}

class JN_MeowVcLoadSpeaker:
    CATEGORY = CATEGORY_AUDIO_MEOW_VC
    RETURN_TYPES = ("MEOW_VC_TARGET_TOKENS",)
    RETURN_NAMES = ("target_tokens",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = sorted(list(set([d for d in glob.iglob("**/**", root_dir=input_dir, recursive=True) if os.path.isfile(os.path.join(input_dir, d)) and d.endswith(".npz")])))
        return {
            "required": {
                "speaker": (files,),
            },
        }

    def run(self, speaker):
        target_tokens = None

        file_path = folder_paths.get_annotated_filepath(speaker)
        speaker = np.load(file_path)

        if speaker:
            if "speaker" in speaker:
                target_tokens = torch.from_numpy(speaker["speaker"])

        return (target_tokens,)

    @classmethod
    def IS_CHANGED(s, speaker):
        file_path = folder_paths.get_annotated_filepath(speaker)
        m = hashlib.sha256()
        with open(file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, speaker):
        if not folder_paths.exists_annotated_filepath(speaker):
            return "Invalid speaker file: {}".format(speaker)

        return True

class JN_MeowSaveVoice:
    CATEGORY = CATEGORY_AUDIO_MEOW
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "semantic_tokens": ("MEOW_TTS_SEMANTIC_TOKENS",),
                "coarse_tokens": ("MEOW_TTS_COARSE_TOKENS",),
                "fine_tokens": ("MEOW_TTS_FINE_TOKENS",),
                "target_tokens": ("MEOW_VC_TARGET_TOKENS",),
                "filename_prefix": ("STRING", {"default": "meow_voice/ComfyUI"}),
                "pitch_semitones": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.5}),
            },
        }

    def run(self, semantic_tokens, coarse_tokens, fine_tokens, target_tokens, filename_prefix, pitch_semitones=0):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        semantic_tokens = [semantic_tokens] if not isinstance(semantic_tokens, list) else semantic_tokens
        coarse_tokens = [coarse_tokens] if not isinstance(coarse_tokens, list) else coarse_tokens
        fine_tokens = [fine_tokens] if not isinstance(fine_tokens, list) else fine_tokens

        for i in range(len(semantic_tokens)):
            file = filename.replace("%batch_num%", str(i))
            file = f"{file}_{counter:05}_.npz"
            file = os.path.join(full_output_folder, file)

            voice = {
                "semantic": semantic_tokens[i],
                "coarse": coarse_tokens[i],
                "fine": fine_tokens[i],
                "speaker": target_tokens,
                "pitch_semitones": pitch_semitones,
            }

            np.savez(file, **voice)

            counter += 1

        return {"ui": {}}

class JN_MeowLoadVoice:
    CATEGORY = CATEGORY_AUDIO_MEOW
    RETURN_TYPES = ("MEOW_TTS_SEMANTIC_TOKENS", "MEOW_TTS_COARSE_TOKENS", "MEOW_TTS_FINE_TOKENS", "MEOW_VC_TARGET_TOKENS", "FLOAT")
    RETURN_NAMES = ("semantic_tokens", "coarse_tokens", "fine_tokens", "target_tokens", "pitch_semitones")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = sorted(list(set([d for d in glob.iglob("**/**", root_dir=input_dir, recursive=True) if os.path.isfile(os.path.join(input_dir, d)) and d.endswith(".npz")])))
        return {
            "required": {
                "voice": (files,),
            },
        }

    def run(self, voice):
        semantic_tokens = None
        coarse_tokens = None
        fine_tokens = None
        target_tokens = None
        pitch_semitones = 0

        file_path = folder_paths.get_annotated_filepath(voice)
        voice = np.load(file_path)

        if voice:
            if "semantic" in voice:
                semantic_tokens = voice["semantic"]

            if "coarse" in voice:
                coarse_tokens = voice["coarse"]

            if "fine" in voice:
                fine_tokens = voice["fine"]

            if "speaker" in voice:
                target_tokens = torch.from_numpy(voice["speaker"])

            if "pitch_semitones" in voice:
                pitch_semitones = float(voice["pitch_semitones"])

        return (semantic_tokens, coarse_tokens, fine_tokens, target_tokens, pitch_semitones)

    @classmethod
    def IS_CHANGED(s, voice):
        file_path = folder_paths.get_annotated_filepath(voice)
        m = hashlib.sha256()
        with open(file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, voice):
        if not folder_paths.exists_annotated_filepath(voice):
            return "Invalid voice file: {}".format(voice)

        return True

class JN_MeowHrtfPosition:
    CATEGORY = CATEGORY_AUDIO_MEOW_HRTF
    RETURN_TYPES = ("MEOW_HRTF_POSITION", "ARRAY")
    RETURN_NAMES = ("position", "positions")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "azimuth_angle": ("INT", {"default": 0, "min": -90, "max": 90}),
                "elevation_angle": ("INT", {"default": 0, "min": 0, "max": 270}),
                "proximity": ("FLOAT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 0.001}),
                "delay_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 0.001}),
            },
            "optional": {
                "positions": ("*", {"multiple": True}),
            },
        }

    def run(self, positions=None, azimuth_angle=0, elevation_angle=0, proximity=1, delay_seconds=0):
        if positions is None:
            positions = []

        positions = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), positions, [None])
        positions = [position for position in positions if position is not None]

        position = {
            "azimuth": azimuth_angle,
            "elevation": elevation_angle,
            "proximity": proximity,
            "delay": delay_seconds,
        }

        positions.append(position)

        return (position, positions)

class JN_MeowHrtfAudio3d:
    CATEGORY = CATEGORY_AUDIO_MEOW_HRTF
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hrtf": ("MEOW_HRTF_MODEL",),
                "audios": ("*", {"multiple": True}),
                "positions": ("*", {"multiple": True}),
            },
        }

    def run(self, hrtf, audios, positions):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        positions = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), positions, [None])
        positions = [position for position in positions if position is not None]

        audios = hrtf.execute(audio=audios, positions=positions)

        return (audios,)

NODE_CLASS_MAPPINGS = {
    "JN_MeowTts": JN_MeowTts,
    "JN_MeowTtsSemantic": JN_MeowTtsSemantic,
    "JN_MeowTtsCoarse": JN_MeowTtsCoarse,
    "JN_MeowTtsFine": JN_MeowTtsFine,
    "JN_MeowTtsDecode": JN_MeowTtsDecode,

    "JN_MeowTtsModel": JN_MeowTtsModel,
    "JN_MeowTtsModelSemantic": JN_MeowTtsModelSemantic,
    "JN_MeowTtsModelCoarse": JN_MeowTtsModelCoarse,
    "JN_MeowTtsModelFine": JN_MeowTtsModelFine,
    "JN_MeowTtsModelEncodec": JN_MeowTtsModelEncodec,
    "JN_MeowTtsModelHubert": JN_MeowTtsModelHubert,
    "JN_MeowTtsTokenizerHubert": JN_MeowTtsTokenizerHubert,

    "JN_MeowTtsSaveContext": JN_MeowTtsSaveContext,
    "JN_MeowTtsLoadContext": JN_MeowTtsLoadContext,
    "JN_MeowTtsAudioToContext": JN_MeowTtsAudioToContext,

    "JN_MeowVc": JN_MeowVc,
    "JN_MeowVcConvertVoice": JN_MeowVcConvertVoice,
    "JN_MeowVcEncodeSource": JN_MeowVcEncodeSource,
    "JN_MeowVcEncodeTarget": JN_MeowVcEncodeTarget,

    "JN_MeowVcModelWavLM": JN_MeowVcModelWavLM,
    "JN_MeowVcModelFreeVC": JN_MeowVcModelFreeVC,

    "JN_MeowVcSaveSpeaker": JN_MeowVcSaveSpeaker,
    "JN_MeowVcLoadSpeaker": JN_MeowVcLoadSpeaker,

    "JN_MeowHrtfAudio3d": JN_MeowHrtfAudio3d,
    "JN_MeowHrtfPosition": JN_MeowHrtfPosition,
    "JN_MeowHrtfModel": JN_MeowHrtfModel,

    "JN_MeowSentenceSplit": JN_MeowSentenceSplit,
    "JN_MeowSaveVoice": JN_MeowSaveVoice,
    "JN_MeowLoadVoice": JN_MeowLoadVoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_MeowTts": "Meow TTS",
    "JN_MeowTtsSemantic": "Meow TTS Semantic",
    "JN_MeowTtsCoarse": "Meow TTS Coarse",
    "JN_MeowTtsFine": "Meow TTS Fine",
    "JN_MeowTtsDecode": "Meow TTS Decode",

    "JN_MeowTtsModel": "Meow TTS Model",
    "JN_MeowTtsModelSemantic": "Meow TTS Model Semantic",
    "JN_MeowTtsModelCoarse": "Meow TTS Model Coarse",
    "JN_MeowTtsModelFine": "Meow TTS Model Fine",
    "JN_MeowTtsModelEncodec": "Meow TTS Model Encodec",
    "JN_MeowTtsModelHubert": "Meow TTS Model Hubert",
    "JN_MeowTtsTokenizerHubert": "Meow TTS Tokenizer Hubert",

    "JN_MeowTtsSaveContext": "Meow TTS Save Context",
    "JN_MeowTtsLoadContext": "Meow TTS Load Context",
    "JN_MeowTtsAudioToContext": "Meow TTS Audio To Context",

    "JN_MeowVc": "Meow Voice Conversion",
    "JN_MeowVcConvertVoice": "Meow VC Convert Voice",
    "JN_MeowVcEncodeSource": "Meow VC Encode Source",
    "JN_MeowVcEncodeTarget": "Meow VC Encode Target",

    "JN_MeowVcModelWavLM": "Meow VC Model WavLM",
    "JN_MeowVcModelFreeVC": "Meow VC Model FreeVC",

    "JN_MeowVcSaveSpeaker": "Meow VC Save Speaker",
    "JN_MeowVcLoadSpeaker": "Meow VC Load Speaker",

    "JN_MeowHrtfAudio3d": "Meow HRTF Audio 3D",
    "JN_MeowHrtfPosition": "Meow HRTF Position",
    "JN_MeowHrtfModel": "Meow HRTF Model",

    "JN_MeowSentenceSplit": "Meow Sentence Split",
    "JN_MeowSaveVoice": "Meow Save Voice",
    "JN_MeowLoadVoice": "Meow Load Voice",
}
