from functools import reduce
import numpy as np
import torch
import torchaudio
import folder_paths
import os
import glob
import hashlib
import io
import json
import random
import librosa
import sox
import psola
import scipy.signal as sig
from PIL import Image
from noisereduce.torchgate import TorchGate as TG
from comfy.cli_args import args
import warnings

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, message=".*Starting a Matplotlib GUI outside of the main thread.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created.*")

from comfy_extras.nodes_audio import insert_or_replace_vorbis_comment

from ..utils import CATEGORY_AUDIO, CATEGORY_AUDIO_CHANNELS, CATEGORY_AUDIO_SAMPLES, CATEGORY_AUDIO_EDIT, DEVICES, get_device

def batch_to_array(audio):
    if audio is None:
        return [None]

    (b, c, s) = audio["waveform"].shape

    audios = []

    base_audio = clone_audio(audio)

    for x in range(0, audio["waveform"].shape[0]):
        c_audio = base_audio.copy()
        c_audio["waveform"] = audio["waveform"][x].clone().reshape(1, c, s)
        audios.append(c_audio)

    return audios

def clone_audio(audio):
    audio_clone = audio.copy()
    audio_clone["waveform"] = audio["waveform"].clone()
    return audio_clone

def audio_merge_channels(audio):
    c_audio = clone_audio(audio)
    np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

    transformer = sox.Transformer()
    transformer.remix(remix_dictionary={1: [x for x in range(1, np_audio.shape[0] + 1)]}, num_output_channels=1)
    transformer.set_output_format(channels=1)
    np_audio = transformer.build_array(input_array=np_audio.transpose(1, 0), sample_rate_in=audio["sample_rate"])

    if np_audio.ndim == 2:
        np_audio = np_audio.transpose(1, 0)
    else:
        np_audio = np.expand_dims(np_audio, axis=0)

    c_audio["waveform"] = torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0)

    return c_audio

def audio_set_channels(audio, channels=1):
    (b, c, s) = audio["waveform"].shape

    if c == channels:
        return clone_audio(audio)

    audio = clone_audio(audio)

    if c < channels:
        if c > 1:
            waveform = audio_merge_channels(audio)["waveform"]
        else:
            waveform = audio["waveform"][:, 0, :].clone().reshape(b, -1, s)

        cat = [audio["waveform"]]

        for _ in range(channels - c):
            cat.append(waveform)

        audio["waveform"] = torch.cat(cat, dim=1).reshape(b, -1, s)
    elif c > channels:
        if channels > 1:
            # channels to keep
            waveform = audio["waveform"][:, 0:channels, :].clone().reshape(b, -1, s)

            c_audio = clone_audio(audio)

            # merge other channels
            c_audio["waveform"] = audio["waveform"][:, channels:, :].clone().reshape(b, -1, s)
            if c_audio["waveform"].shape[1] > 1:
                c_audio = audio_merge_channels(c_audio)

            merge_waveform = c_audio["waveform"]

            # merge other channels on each channel to keep
            for i in range(waveform.shape[1]):
                c_audio["waveform"] = torch.cat((
                    waveform[:, i, :].clone().reshape(b, -1, s),
                    merge_waveform
                ), dim=1)
                c_audio = audio_merge_channels(c_audio)
                waveform[:, i, :] = c_audio["waveform"]

            c_audio["waveform"] = waveform

            audio = c_audio
        else:
            audio = audio_merge_channels(audio)

    return audio

def audio_set_samples(audio, samples):
    (b, c, s) = audio["waveform"].shape

    if s == samples:
        return clone_audio(audio)

    if s < samples:
        silence = torch.zeros((b, c, samples - s), dtype=audio["waveform"].dtype)
        c_audio = clone_audio(audio)
        c_audio["waveform"] = torch.cat((c_audio["waveform"], silence), dim=2)
        audio = c_audio
    elif s > samples:
        c_audio = clone_audio(audio)
        c_audio["waveform"] = c_audio["waveform"][:, :, :samples].clone().reshape(b, c, samples)
        audio = c_audio

    return audio

def audio_resample(audio, sample_rate, quality="soxr_vhq"):
    if audio["sample_rate"] == sample_rate:
        return clone_audio(audio)

    c_audio = clone_audio(audio)
    np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

    np_audio = librosa.resample(np_audio, orig_sr=audio["sample_rate"], target_sr=sample_rate, res_type=quality)

    if np_audio.ndim == 1:
        np_audio = np.expand_dims(np_audio, axis=0)

    c_audio["sample_rate"] = sample_rate
    c_audio["waveform"] = torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0)

    return c_audio

def audios_stack_channels(audios):
    audio_batch = None

    sample_rate = 1
    for audio in audios:
        sample_rate = max(sample_rate, audio["sample_rate"])

    samples = 1
    new_audios = []
    for audio in audios:
        audio = audio_resample(audio, sample_rate=sample_rate)
        samples = max(samples, audio["waveform"].shape[2])
        new_audios.append(audio)
    audios = new_audios

    waveforms = []

    for audio in audios:
        audio = audio_set_samples(audio, samples=samples)

        if audio_batch is None:
            audio_batch = audio

        waveforms.append(audio["waveform"])

    if audio_batch and waveforms:
        audio_batch["waveform"] = torch.cat(waveforms, dim=1)

    return audio_batch

def audios_concatenate(audios, silence_seconds=0):
    audio_batch = None
    channels = None

    for audio in audios:
        if audio_batch is None:
            channels = audio["waveform"].shape[1]
            audio_batch = clone_audio(audio)
        else:
            channels = max(channels, audio["waveform"].shape[1])
            audio_batch = audio_set_channels(audio_batch, channels)
            audio = audio_set_channels(audio, channels)
            cat = [audio_batch["waveform"]]
            if silence_seconds > 0:
                silence = torch.zeros((1, channels, int(audio_batch["sample_rate"] * silence_seconds)), dtype=audio_batch["waveform"].dtype)
                cat.append(silence)
            cat.append(audio["waveform"])
            audio_batch["waveform"] = torch.cat(cat, dim=2)

    return audio_batch

class JN_SaveAudio:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
            },
            "optional": {
                "audioUI": ("AUDIO_UI",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, audios, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        return self.save_audio(audios=audios, filename_prefix=filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)

    def save_audio(self, audios, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        results = []

        filename_prefix += self.prefix_append

        batch_number = 0

        for audio in audios:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

            metadata = {}
            if not args.disable_metadata:
                if prompt is not None:
                    metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata[x] = json.dumps(extra_pnginfo[x])

            for waveform in audio["waveform"].cpu():
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.flac"

                buff = io.BytesIO()
                torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

                buff = insert_or_replace_vorbis_comment(buff, metadata)

                with open(os.path.join(full_output_folder, file), 'wb') as f:
                    f.write(buff.getbuffer())

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
                batch_number += 1

        return { "ui": { "audio": results } }

class JN_PreviewAudio(JN_SaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
            },
            "optional": {
                "audioUI": ("AUDIO_UI",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

class JN_LoadAudioDirectory:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif')

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        dirs = sorted(list(set([d.rstrip(os.sep) for d in glob.iglob("**/**", root_dir=input_dir, recursive=True) if os.path.isdir(os.path.join(input_dir, d))])))
        return {
            "required": {
                "directory": (dirs,),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": True}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, directory, recursive=True, limit=0, offset=0):
        directory_path = folder_paths.get_annotated_filepath(directory)
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(self.SUPPORTED_FORMATS)])

        if limit > 0:
            files = files[offset:offset+limit]

        audios = []

        for file in files:
            audio  = self.load_audio(audio=file)

            audios.append(audio)

        return (audios,)

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return audio

    @classmethod
    def IS_CHANGED(s, directory, recursive=True, limit=0, offset=0):
        directory_path = folder_paths.get_annotated_filepath(directory)
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(s.SUPPORTED_FORMATS)])

        if limit > 0:
            files = files[offset:offset+limit]

        m = hashlib.sha256()

        for file in files:
            audio_path = folder_paths.get_annotated_filepath(file)
            with open(audio_path, 'rb') as f:
                m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory, **kwargs):
        if not folder_paths.exists_annotated_filepath(directory):
            return "Invalid directory: {}".format(directory)

        return True

class JN_AudioConcatenation:
    CATEGORY = CATEGORY_AUDIO_SAMPLES
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
            },
            "optional": {
                "silence_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
            },
        }

    def run(self, audios, silence_seconds=0):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        audio_batch = audios_concatenate(audios, silence_seconds=silence_seconds)

        return (audio_batch,)

class JN_AudioSlice:
    CATEGORY = CATEGORY_AUDIO_SAMPLES
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    OPERATIONS = ["[a:b]", "[a:a+b]", "[a-b:a]", "[a-b:a+b]", "[a:]", "[:a]"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "a_seconds": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01}),
                "b_seconds": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01}),
                "operation": (s.OPERATIONS,),
            },
        }

    def run(self, audios, a_seconds, b_seconds, operation):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        new_audios = []

        for audio in audios:
            sample_rate = audio["sample_rate"]

            a = round(sample_rate * a_seconds)
            b = round(sample_rate * b_seconds)

            c_audio = clone_audio(audio)

            if operation == "[a:b]":
                c_audio["waveform"] = c_audio["waveform"][:, :, a:b]
            elif operation == "[a:a+b]":
                c_audio["waveform"] = c_audio["waveform"][:, :, a:a+b]
            elif operation == "[a-b:a]":
                c_audio["waveform"] = c_audio["waveform"][:, :, a-b:a]
            elif operation == "[a-b:a+b]":
                c_audio["waveform"] = c_audio["waveform"][:, :, a-b:a+b]
            elif operation == "[a:]":
                c_audio["waveform"] = c_audio["waveform"][:, :, a:]
            elif operation == "[:a]":
                c_audio["waveform"] = c_audio["waveform"][:, :, :a]

            new_audios.append(c_audio)

        return (new_audios,)

class JN_AudioStackChannels:
    CATEGORY = CATEGORY_AUDIO_CHANNELS
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
            },
        }

    def run(self, audios):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        audio_batch = audios_stack_channels(audios)

        return (audio_batch,)

class JN_AudioSetChannels:
    CATEGORY = CATEGORY_AUDIO_CHANNELS
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "channels": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, audios, channels=1):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        output = []

        for audio in audios:
            output.append(audio_set_channels(audio, channels))

        return (output,)

class JN_AudioGetChannels:
    CATEGORY = CATEGORY_AUDIO_CHANNELS
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "from_channel": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "to_channel": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, audios, from_channel=0, to_channel=0):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        min_channel = min(from_channel, to_channel)
        max_channel = max(from_channel, to_channel)

        output = []

        for audio in audios:
            (b, c, s) = audio["waveform"].shape
            from_channel = min(min_channel, c-1)
            to_channel = min(max_channel, c-1) + 1

            c_audio = clone_audio(audio)
            c_audio["waveform"] = c_audio["waveform"][:, from_channel:to_channel, :].reshape(b, -1, s)

            output.append(c_audio)

        return (output,)

class JN_AudioBatchToArray:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
            },
        }

    def run(self, audios):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        return (audios,)

class JN_AudioArrayToBatch:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
            },
        }

    def run(self, audios):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]

        channels = 1
        samples = 1
        sample_rate = 1

        for audio in audios:
            channels = max(channels, audio["waveform"].shape[1])
            sample_rate = max(sample_rate, audio["sample_rate"])

        new_audios = []
        for audio in audios:
            audio = audio_resample(audio, sample_rate=sample_rate)
            audio = audio_set_channels(audio, channels)
            samples = max(samples, audio["waveform"].shape[2])
            new_audios.append(audio)
        audios = new_audios

        waveforms = []
        for audio in audios:
            audio = audio_set_samples(audio, samples)
            waveforms.append(audio["waveform"])

        audio = {
            "sample_rate": sample_rate,
            "waveform": torch.cat(waveforms, dim=0)
        }

        return (audio,)

class JN_AudioTrimSilence:
    CATEGORY = CATEGORY_AUDIO_SAMPLES
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    METHODS = [
        "simple",
        "aggressive",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "method": (s.METHODS,),
                "top_db": ("INT", {"default": 30, "min": 0, "max": 0xffffffffffffffff}),
                "sound_limit": ("FLOAT", {"default": 0.005, "min": 0, "max": 0xffffffffffffffff, "step": 0.001}),
                "delimiter_seconds": ("FLOAT", {"default": 0.15, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                "max_silence_seconds": ("FLOAT", {"default": 0.35, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
            },
        }

    def run(self, audios, method, top_db=30, sound_limit=0.005, delimiter_seconds=0.15, max_silence_seconds=0.35):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        output = []

        for audio in audios:
            c_audio = clone_audio(audio)
            np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

            if method == "aggressive":
                np_audio = self.trim_noise(np_audio, sample_rate=c_audio["sample_rate"], sound_limit=sound_limit, delimiter_seconds=delimiter_seconds, max_silence_seconds=max_silence_seconds)

            np_audio, _ = librosa.effects.trim(y=np_audio, top_db=30)

            c_audio["waveform"] = torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0)

            output.append(c_audio)

        return (output,)

    def trim_noise(self, audio, sample_rate, sound_limit=0.005, delimiter_seconds=0.15, max_silence_seconds=0.35):
        audio_array = audio.copy()
        limit_silence = round(sample_rate * max_silence_seconds)

        # end

        count = round(sample_rate * delimiter_seconds)
        to_zero = False
        size_silence = 0

        for x in range(audio_array.shape[1]-1, -1, -1):
            if np.max(np.abs(audio_array[:, x:x+1])) > sound_limit:
                to_zero = True
            else:
                if count <= 0:
                    break
                count = count - 1

            if to_zero:
                size_silence += 1

        if size_silence > 0 and size_silence <= limit_silence:
            audio_array[:, -size_silence:] = np.zeros((audio_array.shape[0], size_silence))

        # start

        count = round(sample_rate * delimiter_seconds)
        to_zero = False
        size_silence = 0

        for x in range(audio_array.shape[1]):
            if np.max(np.abs(audio_array[:, x:x+1])) > sound_limit:
                to_zero = True
            else:
                if count <= 0:
                    break
                count = count - 1

            if to_zero:
                size_silence += 1

        if size_silence > 0 and size_silence <= limit_silence:
            audio_array[:, :size_silence] = np.zeros((audio_array.shape[0], size_silence))

        return audio_array

class JN_AudioNoiseReduction:
    CATEGORY = CATEGORY_AUDIO_EDIT
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "device": (DEVICES,),
                "stationary": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    def run(self, audios, device, stationary=False, strength=1.0):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        device = get_device(device)

        tgs = {}

        output = []

        for audio in audios:
            sample_rate = audio["sample_rate"]

            if sample_rate in tgs:
                tg = tgs[sample_rate]
            else:
                tg = TG(sr=sample_rate, nonstationary=not stationary, prop_decrease=strength).to(device)
                tgs[sample_rate] = tg

            c_audio = clone_audio(audio)
            old_device = audio["waveform"].device
            c_audio["waveform"] = tg(audio["waveform"].squeeze(0).to(device)).unsqueeze(0).to(old_device)

            output.append(c_audio)

        return (output,)

class JN_AudioSoxTransformerBase:
    CATEGORY = CATEGORY_AUDIO_EDIT
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    def run(self, audios, **kwargs):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        transformer = sox.Transformer()
        transformer = self.build_transformer(transformer=transformer, **kwargs)

        output = []

        for audio in audios:
            c_audio = clone_audio(audio)
            np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

            np_audio = transformer.build_array(input_array=np_audio.transpose(1, 0), sample_rate_in=audio["sample_rate"])

            if np_audio.ndim == 2:
                np_audio = np_audio.transpose(1, 0)
            else:
                np_audio = np.expand_dims(np_audio, axis=0)

            c_audio["waveform"] = torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0)

            output.append(c_audio)

        return (output,)

    def build_transformer(self, *args, **kwargs):
        pass

class JN_AudioNormalize(JN_AudioSoxTransformerBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "db_level": ("FLOAT", {"default": -3, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.1}),
            },
        }

    def build_transformer(self, transformer, db_level=-3.0):
        transformer.norm(db_level=db_level)
        return transformer

class JN_AudioSpeed(JN_AudioSoxTransformerBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
            },
        }

    def build_transformer(self, transformer, factor=1.0):
        transformer.speed(factor=factor)
        return transformer

class JN_AudioTempo(JN_AudioSoxTransformerBase):
    METHODS = {
        "normal": None,
        "music": "m",
        "speech": "s",
        "linear": "l",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 0.05}),
                "method": (list(s.METHODS.keys()),),
                "quick": ("BOOLEAN", {"default": False}),
            },
        }

    def build_transformer(self, transformer, factor=1.0, method="normal", quick=False):
        audio_type = self.METHODS[method]
        transformer.tempo(factor=factor, audio_type=audio_type, quick=quick)
        return transformer

class JN_AudioVolume(JN_AudioSoxTransformerBase):
    GAIN_TYPES = [
        "amplitude",
        "power",
        "db",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "gain": ("FLOAT", {"default": 1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "gain_type": (s.GAIN_TYPES,),
                "limiter_gain": ("FLOAT", {"default": 0, "min": 0, "max": 0.999, "step": 0.001}),
            },
        }

    def build_transformer(self, transformer, gain=0.0, gain_type="amplitude", limiter_gain=0):
        if limiter_gain <= 0:
            limiter_gain = None

        if gain_type in ["amplitude", "power"] and gain < 0:
            gain = 0

        transformer.vol(gain=gain, gain_type=gain_type, limiter_gain=limiter_gain)
        return transformer

class JN_AudioPitch(JN_AudioSoxTransformerBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "semitones": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.5}),
                "quick": ("BOOLEAN", {"default": False}),
            },
        }

    def build_transformer(self, transformer, semitones=0.0, quick=False):
        transformer.pitch(n_semitones=semitones, quick=quick)
        return transformer

class JN_AudioReverberation(JN_AudioSoxTransformerBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "reverberance_percentage": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.5}),
                "room_scale_percentage": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 0.5}),
                "high_freq_damping_percentage": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.5}),
                "stereo_depth_percentage": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 0.5}),
                "pre_delay_milliseconds": ("FLOAT", {"default": 20, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "wet_gain_db": ("FLOAT", {"default": -10, "min": -10, "max": 10, "step": 0.5}),
                "wet_only": ("BOOLEAN", {"default": False}),
            },
        }

    def build_transformer(self, transformer,
            reverberance_percentage=50.0,
            high_freq_damping_percentage=50.0,
            room_scale_percentage=100,
            stereo_depth_percentage=100,
            pre_delay_milliseconds=0,
            wet_gain_db=0,
            wet_only=False):
        transformer.reverb(
            reverberance=reverberance_percentage,
            high_freq_damping=high_freq_damping_percentage,
            room_scale=room_scale_percentage,
            stereo_depth=stereo_depth_percentage,
            pre_delay=pre_delay_milliseconds,
            wet_gain=wet_gain_db,
            wet_only=wet_only,
        )
        return transformer

class JN_AudioSampleRate():
    CATEGORY = CATEGORY_AUDIO_EDIT
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    QUALITIES = {
        "very high": "soxr_vhq",
        "high": "soxr_hq",
        "medium": "soxr_mq",
        "low": "soxr_lq",
        "very low": "soxr_qq",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "sample_rate": ("INT", {"default": 24000, "min": 1, "max": 0xffffffffffffffff}),
                "quality": (list(s.QUALITIES.keys()),),
            },
        }

    def run(self, audios, sample_rate=24000, quality="very high"):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        quality = self.QUALITIES[quality]

        output = []

        for audio in audios:
            output.append(audio_resample(audio, sample_rate=sample_rate, quality=quality))

        return (output,)

class JN_AudioAutoTune:
    CATEGORY = CATEGORY_AUDIO_EDIT
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("audios",)
    FUNCTION = "run"

    SCALE_TONICS = ["C", "D", "E", "F", "G", "A", "B"]
    SCALE_ACCIDENTALS = ["", "b", "#"]
    SCALE_KEYS = ["maj", "min", "ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]

    SEMITONES_IN_OCTAVE = 12

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "auto": ("BOOLEAN", {"default": True}),
                "scale_tonic": (s.SCALE_TONICS,),
                "scale_accidental": (s.SCALE_ACCIDENTALS,),
                "scale_key": (s.SCALE_KEYS,),
            },
        }

    def run(self, audios, scale_tonic, scale_accidental, scale_key, auto=True):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        if auto:
            scale = "auto"
        else:
            scale = f"{scale_tonic}{scale_accidental}:{scale_key}"

        output = []

        for audio in audios:
            c_audio = clone_audio(audio)
            np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

            channels = []
            for channel in range(np_audio.shape[0]):
                channels.append(self.autotune(np_audio[channel], sample_rate=audio["sample_rate"], scale=scale))
            np_audio = np.stack(channels, axis=0)

            if np_audio.ndim == 1:
                np_audio = np.expand_dims(np_audio, axis=0)

            c_audio["waveform"] = torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0)

            output.append(c_audio)

        return (output,)

    def autotune(self, audio, sample_rate, scale="auto"):
        # Set some basis parameters.
        frame_length = 2048
        hop_length = frame_length // 4
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')

        # Pitch tracking using the PYIN algorithm.
        f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                            frame_length=frame_length,
                                                            hop_length=hop_length,
                                                            sr=sample_rate,
                                                            fmin=fmin,
                                                            fmax=fmax)

        # Apply the chosen adjustment strategy to the pitch.
        if scale == "auto":
            corrected_f0 = self.closest_pitch(f0)
        else:
            corrected_f0 = self.closest_pitch_from_scale(f0, scale)

        # Pitch-shifting using the PSOLA algorithm.
        return psola.vocode(audio, sample_rate=int(sample_rate), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

    def closest_pitch(self, f0):
        """Round the given pitch values to the nearest MIDI note numbers"""
        midi_note = np.around(librosa.hz_to_midi(f0))
        # To preserve the nan values.
        nan_indices = np.isnan(f0)
        midi_note[nan_indices] = np.nan
        # Convert back to Hz.
        return librosa.midi_to_hz(midi_note)

    def closest_pitch_from_scale(self, f0, scale):
        """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
        sanitized_pitch = np.zeros_like(f0)
        for i in np.arange(f0.shape[0]):
            sanitized_pitch[i] = self.closest_pitch_from_scale2(f0[i], scale)
        # Perform median filtering to additionally smooth the corrected pitch.
        smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
        # Remove the additional NaN values after median filtering.
        smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
            sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
        return smoothed_sanitized_pitch

    def closest_pitch_from_scale2(self, f0, scale):
        """Return the pitch closest to f0 that belongs to the given scale"""
        # Preserve nan.
        if np.isnan(f0):
            return np.nan
        degrees = self.degrees_from(scale)
        midi_note = librosa.hz_to_midi(f0)
        # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
        # input pitch.
        degree = midi_note % self.SEMITONES_IN_OCTAVE
        # Find the closest pitch class from the scale.
        degree_id = np.argmin(np.abs(degrees - degree))
        # Calculate the difference between the input pitch class and the desired pitch class.
        degree_difference = degree - degrees[degree_id]
        # Shift the input MIDI note number by the calculated difference.
        midi_note -= degree_difference
        # Convert to Hz.
        return librosa.midi_to_hz(midi_note)

    def degrees_from(self, scale):
        """Return the pitch classes (degrees) that correspond to the given scale"""
        degrees = librosa.key_to_degrees(scale)
        # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
        # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
        # would be incorrectly assigned.
        degrees = np.concatenate((degrees, [degrees[0] + self.SEMITONES_IN_OCTAVE]))
        return degrees

class JN_AudioPlot:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    PLOTS = [
        "waveform",
        "pitch",
        "loudness",
        "strength",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),
                "plot": (s.PLOTS,),
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "width": ("INT", {"default": 1500, "min": 1, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 1000, "min": 1, "max": 0xffffffffffffffff}),
                "single_image": ("BOOLEAN", {"default": False}),
                "plot_difference": ("BOOLEAN", {"default": False}),
                "plot_difference_absolute": ("BOOLEAN", {"default": False}),
                "colors": ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
                "labels": ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
            },
        }

    def run(self, audios, plot, title="", width=2000, height=1000, single_image=False, plot_difference=False, plot_difference_absolute=False, colors="", labels=""):
        audios = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), audios, [None])
        audios = [audio for audio in audios if audio is not None]
        audios = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), audios, [])

        colors = [color.strip() for color in ",".join(colors.split("\n")).split(",")]
        colors = [color for color in colors if color != ""]

        labels = [label.strip() for label in labels.split("\n")]
        labels = [label for label in labels if label != ""]

        def get_color(idx):
            if idx >= len(colors):
                return None
            return colors[idx % len(colors)]

        def get_label(idx):
            if idx >= len(labels):
                return f"Audio {idx}"
            return labels[idx % len(labels)]

        if len(audios) <= 1:
            plot_difference = False
            plot_difference_absolute = False

        images = []

        if title:
            title_prefix = f"{title} "
        else:
            title_prefix = ""

        num_plots = 1
        plots_per_feature = 1
        idx = 0

        if single_image:
            plt.figure(figsize=(width * 0.01, height * 0.01))

            num_plots = 1
            plots_per_feature = 1
            idx = 0

            if plot_difference:
                plots_per_feature += 1
            if plot_difference_absolute:
                plots_per_feature += 1

            channels = 1
            samples = 1
            sample_rate = 1

            for audio in audios:
                channels = max(channels, audio["waveform"].shape[1])
                sample_rate = max(sample_rate, audio["sample_rate"])

            new_audios = []
            for audio in audios:
                audio = audio_resample(audio, sample_rate=sample_rate)
                audio = audio_set_channels(audio, channels)
                samples = max(samples, audio["waveform"].shape[2])
                new_audios.append(audio)
            audios = new_audios

            new_audios = []
            for audio in audios:
                audio = audio_set_samples(audio, samples)
                new_audios.append(audio)
            audios = new_audios

            plots_per_feature *= channels

            plt.figure(figsize=(width * 0.01, height * 0.01))

            n_plot = 1

            for channel in range(channels):
                plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + n_plot)

                for audio_idx, audio in enumerate(audios):
                    c_audio = clone_audio(audio)
                    sample_rate = audio["sample_rate"]
                    np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                    if plot == "waveform":
                        time = np.arange(np_audio.shape[1]) / sample_rate
                        plt.plot(time, np_audio[channel], label=get_label(audio_idx), color=get_color(audio_idx))

                    elif plot == "pitch":
                        _, pitch = librosa.effects.hpss(np_audio[channel])
                        time = np.arange(np_audio.shape[1]) / sample_rate
                        plt.plot(time, pitch, label=get_label(audio_idx), color=get_color(audio_idx))

                    elif plot == "loudness":
                        loudness = librosa.feature.rms(y=np_audio[channel])[0]
                        time = librosa.times_like(loudness, sr=sample_rate)
                        plt.plot(time, loudness, label=get_label(audio_idx), color=get_color(audio_idx))

                    elif plot == "strength":
                        strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                        time = librosa.times_like(strength, sr=sample_rate)
                        plt.plot(time, strength, label=get_label(audio_idx), color=get_color(audio_idx))

                plt.xlabel("Time (s)")

                if plot == "waveform":
                    plt.ylabel("Amplitude")
                    plt.title(f"{title_prefix}[Waveform Channel {channel}]")
                    plt.legend()
                    plt.grid(True)

                elif plot == "pitch":
                    plt.ylabel("Pitch")
                    plt.title(f"{title_prefix}[Pitch Channel {channel}]")
                    plt.legend()
                    plt.grid(True)

                elif plot == "loudness":
                    plt.ylabel("Loudness")
                    plt.title(f"{title_prefix}[Loudness Channel {channel}]")
                    plt.legend()
                    plt.grid(True)

                elif plot == "strength":
                    plt.ylabel("Strength")
                    plt.title(f"{title_prefix}[Strength Channel {channel}]")
                    plt.legend()
                    plt.grid(True)

            if plot_difference:
                n_plot += channels

                for channel in range(channels):
                    plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + n_plot)

                    first_data = None

                    for audio_idx, audio in enumerate(audios):
                        c_audio = clone_audio(audio)
                        sample_rate = audio["sample_rate"]
                        np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                        if plot == "waveform":
                            if audio_idx == 0:
                                first_data = np_audio[channel]
                            else:
                                time = np.arange(np_audio.shape[1]) / sample_rate
                                plt.plot(time, first_data - np_audio[channel], label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "pitch":
                            _, pitch = librosa.effects.hpss(np_audio[channel])
                            if audio_idx == 0:
                                first_data = pitch
                            else:
                                time = np.arange(np_audio.shape[1]) / sample_rate
                                plt.plot(time, first_data - pitch, label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "loudness":
                            loudness = librosa.feature.rms(y=np_audio[channel])[0]
                            if audio_idx == 0:
                                first_data = loudness
                            else:
                                time = librosa.times_like(loudness, sr=sample_rate)
                                plt.plot(time, first_data - loudness, label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "strength":
                            strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                            if audio_idx == 0:
                                first_data = strength
                            else:
                                time = librosa.times_like(strength, sr=sample_rate)
                                plt.plot(time, first_data - strength, label=get_label(audio_idx), color=get_color(audio_idx))

                    plt.xlabel("Time (s)")

                    if plot == "waveform":
                        plt.ylabel("Amplitude")
                        plt.title(f"{title_prefix}[Waveform Channel {channel}] [Difference - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "pitch":
                        plt.ylabel("Pitch")
                        plt.title(f"{title_prefix}[Pitch Channel {channel}] [Difference - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "loudness":
                        plt.ylabel("Loudness")
                        plt.title(f"{title_prefix}[Loudness Channel {channel}] [Difference - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "strength":
                        plt.ylabel("Strength")
                        plt.title(f"{title_prefix}[Strength Channel {channel}] [Difference - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

            if plot_difference_absolute:
                n_plot += channels

                for channel in range(channels):
                    plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + n_plot)

                    first_data = None

                    for audio_idx, audio in enumerate(audios):
                        c_audio = clone_audio(audio)
                        sample_rate = audio["sample_rate"]
                        np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                        time = np.arange(np_audio.shape[1]) / sample_rate

                        if plot == "waveform":
                            if audio_idx == 0:
                                first_data = np_audio[channel]
                            else:
                                plt.plot(time, np.abs(first_data - np_audio[channel]), label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "pitch":
                            _, pitch = librosa.effects.hpss(np_audio[channel])
                            if audio_idx == 0:
                                first_data = pitch
                            else:
                                plt.plot(time, np.abs(first_data - pitch), label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "loudness":
                            loudness = librosa.feature.rms(y=np_audio[channel])[0]
                            if audio_idx == 0:
                                first_data = loudness
                            else:
                                time = librosa.times_like(loudness, sr=sample_rate)
                                plt.plot(time, np.abs(first_data - loudness), label=get_label(audio_idx), color=get_color(audio_idx))

                        elif plot == "strength":
                            strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                            if audio_idx == 0:
                                first_data = strength
                            else:
                                time = librosa.times_like(strength, sr=sample_rate)
                                plt.plot(time, np.abs(first_data - strength), label=get_label(audio_idx), color=get_color(audio_idx))

                    plt.xlabel("Time (s)")

                    if plot == "waveform":
                        plt.ylabel("Amplitude")
                        plt.title(f"{title_prefix}[Waveform Channel {channel}] [Difference Absolute - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "pitch":
                        plt.ylabel("Pitch")
                        plt.title(f"{title_prefix}[Pitch Channel {channel}] [Difference Absolute - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "loudness":
                        plt.ylabel("Loudness")
                        plt.title(f"{title_prefix}[Loudness Channel {channel}] [Difference Absolute - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

                    elif plot == "strength":
                        plt.ylabel("Strength")
                        plt.title(f"{title_prefix}[Strength Channel {channel}] [Difference Absolute - {get_label(0)}]")
                        plt.legend()
                        plt.grid(True)

            plt.tight_layout()
            images.append(self.plt_to_image())

        else:
            for audio_idx, audio in enumerate(audios):
                c_audio = clone_audio(audio)
                sample_rate = audio["sample_rate"]
                np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                num_plots = 1
                plots_per_feature = np_audio.shape[0]
                idx = 0

                plt.figure(figsize=(width * 0.01, height * 0.01))

                if plot == "waveform":
                    for channel in range(np_audio.shape[0]):
                        plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + 1)
                        time = np.arange(np_audio.shape[1]) / sample_rate
                        plt.plot(time, np_audio[channel], label=get_label(audio_idx), color=get_color(audio_idx))
                        plt.xlabel("Time (s)")
                        plt.ylabel("Amplitude")
                        plt.title(f"{title_prefix}[Waveform Channel {channel}]")
                        plt.legend()
                        plt.grid(True)

                elif plot == "pitch":
                    for channel in range(np_audio.shape[0]):
                        _, pitch = librosa.effects.hpss(np_audio[channel])
                        plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + 1)
                        time = np.arange(np_audio.shape[1]) / sample_rate
                        plt.plot(time, pitch, label=get_label(audio_idx), color=get_color(audio_idx))
                        plt.xlabel("Time (s)")
                        plt.ylabel("Pitch")
                        plt.title(f"{title_prefix}[Pitch Channel {channel}]")
                        plt.legend()
                        plt.grid(True)

                elif plot == "loudness":
                    for channel in range(np_audio.shape[0]):
                        loudness = librosa.feature.rms(y=np_audio[channel])[0]
                        plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + 1)
                        time = librosa.times_like(loudness, sr=sample_rate)
                        plt.plot(time, loudness, label=get_label(audio_idx), color=get_color(audio_idx))
                        plt.xlabel("Time (s)")
                        plt.ylabel("Loudness")
                        plt.title(f"{title_prefix}[Loudness Channel {channel}]")
                        plt.legend()
                        plt.grid(True)

                elif plot == "strength":
                    for channel in range(np_audio.shape[0]):
                        strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                        plt.subplot(num_plots * plots_per_feature, 1, idx * plots_per_feature + channel + 1)
                        time = librosa.times_like(strength, sr=sample_rate)
                        plt.plot(time, strength, label=get_label(audio_idx), color=get_color(audio_idx))
                        plt.xlabel("Time (s)")
                        plt.ylabel("Strength")
                        plt.title(f"{title_prefix}[Strength Channel {channel}]")
                        plt.legend()
                        plt.grid(True)

                plt.tight_layout()
                images.append(self.plt_to_image())

        plt.clf()
        plt.close("all")

        return (images,)

    def plt_to_image(plot):
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        i = Image.open(buf)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        buf.close()

        return image

class JN_AudioCompare:
    CATEGORY = CATEGORY_AUDIO
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "FLOAT", "ARRAY")
    RETURN_NAMES = ("equal", "different", "difference", "differences")
    FUNCTION = "run"

    AGGREGATORS = [
        "average",
        "sum",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audios": ("*", {"multiple": True}),

                "difference_limit": ("FLOAT", {"default": 250, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),

                "differences_aggregator": (s.AGGREGATORS,),
                "features_aggregator": (s.AGGREGATORS,),

                "waveform": ("BOOLEAN", {"default": False}),
                "pitch": ("BOOLEAN", {"default": False}),
                "loudness": ("BOOLEAN", {"default": False}),
                "strength": ("BOOLEAN", {"default": False}),

                "waveform_weight": ("FLOAT", {"default": 0.01, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                "pitch_weight": ("FLOAT", {"default": 0.05, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                "loudness_weight": ("FLOAT", {"default": 10, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                "strength_weight": ("FLOAT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
            },
        }

    def run(self, audios, difference_limit=0,
            differences_aggregator="average", features_aggregator="average",
            waveform=False, waveform_weight=0,
            pitch=False, pitch_weight=0,
            loudness=False, loudness_weight=0,
            strength=False, strength_weight=0):
        audios = [JN_AudioArrayToBatch().run(audios=audios)[0]]
        audios = JN_AudioBatchToArray().run(audios=audios)[0]

        differences = []

        features = {
            "waveform": [],
            "pitch": [],
            "loudness": [],
            "strength": [],
        }

        for audio_idx, audio in enumerate(audios):
            if audio_idx == 0:
                c_audio = clone_audio(audio)
                sample_rate = audio["sample_rate"]
                np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                # Save features of the target
                for channel in range(np_audio.shape[0]):
                    if waveform:
                        features["waveform"].append(np_audio[channel])
                    if pitch:
                        _, audio_pitch = librosa.effects.hpss(np_audio[channel])
                        features["pitch"].append(audio_pitch)
                    if loudness:
                        audio_loudness = librosa.feature.rms(y=np_audio[channel])[0]
                        features["loudness"].append(audio_loudness)
                    if strength:
                        audio_strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                        features["strength"].append(audio_strength)
            else:
                # Compare with the target
                if waveform or pitch or loudness or strength:
                    c_audio = clone_audio(audio)
                    sample_rate = audio["sample_rate"]
                    np_audio = c_audio["waveform"].cpu().squeeze(0).numpy()

                    audio_diff = []

                    for channel in range(np_audio.shape[0]):
                        channel_diff = []

                        if waveform:
                            channel_diff.append(waveform_weight * np.sum(np.abs(features["waveform"][channel] - np_audio[channel])))
                        if pitch:
                            _, audio_pitch = librosa.effects.hpss(np_audio[channel])
                            channel_diff.append(pitch_weight * np.sum(np.abs(features["pitch"][channel] - audio_pitch)))
                        if loudness:
                            audio_loudness = librosa.feature.rms(y=np_audio[channel])[0]
                            channel_diff.append(loudness_weight * np.sum(np.abs(features["loudness"][channel] - audio_loudness)))
                        if strength:
                            audio_strength = librosa.onset.onset_strength(y=np_audio[channel], sr=sample_rate)
                            channel_diff.append(strength_weight * np.sum(np.abs(features["strength"][channel] - audio_strength)))

                        if features_aggregator == "sum":
                            channel_diff = np.sum(channel_diff)
                        else:
                            channel_diff = np.average(channel_diff)

                        audio_diff.append(channel_diff)

                    audio_diff = np.average(audio_diff)
                else:
                    audio_diff = np.inf

                differences.append(audio_diff)

        if len(differences) > 0:
            if differences_aggregator == "sum":
                difference = np.sum(differences)
            else:
                difference = np.average(differences)
        else:
            difference = np.inf

        equal = difference <= difference_limit

        return (equal, not equal, difference, differences)

NODE_CLASS_MAPPINGS = {
    "JN_SaveAudio": JN_SaveAudio,
    "JN_PreviewAudio": JN_PreviewAudio,
    "JN_LoadAudioDirectory": JN_LoadAudioDirectory,
    "JN_AudioPlot": JN_AudioPlot,
    "JN_AudioCompare": JN_AudioCompare,

    "JN_AudioBatchToArray": JN_AudioBatchToArray,
    "JN_AudioArrayToBatch": JN_AudioArrayToBatch,

    "JN_AudioSetChannels": JN_AudioSetChannels,
    "JN_AudioGetChannels": JN_AudioGetChannels,
    "JN_AudioStackChannels": JN_AudioStackChannels,

    "JN_AudioConcatenation": JN_AudioConcatenation,
    "JN_AudioSlice": JN_AudioSlice,
    "JN_AudioTrimSilence": JN_AudioTrimSilence,

    "JN_AudioNoiseReduction": JN_AudioNoiseReduction,
    "JN_AudioAutoTune": JN_AudioAutoTune,

    "JN_AudioSampleRate": JN_AudioSampleRate,
    "JN_AudioNormalize": JN_AudioNormalize,
    "JN_AudioSpeed": JN_AudioSpeed,
    "JN_AudioTempo": JN_AudioTempo,
    "JN_AudioVolume": JN_AudioVolume,
    "JN_AudioPitch": JN_AudioPitch,
    "JN_AudioReverberation": JN_AudioReverberation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_SaveAudio": "Save Audio",
    "JN_PreviewAudio": "Preview Audio",
    "JN_LoadAudioDirectory": "Load Audio Directory",
    "JN_AudioPlot": "Audio Plot",
    "JN_AudioCompare": "Audio Compare",

    "JN_AudioBatchToArray": "Audio Batch To Array",
    "JN_AudioArrayToBatch": "Audio Array To Batch",

    "JN_AudioSetChannels": "Audio Set Channels",
    "JN_AudioGetChannels": "Audio Get Channels",
    "JN_AudioStackChannels": "Audio Stack Channels",

    "JN_AudioConcatenation": "Audio Concatenation",
    "JN_AudioSlice": "Audio Slice",
    "JN_AudioTrimSilence": "Audio Trim Silence",

    "JN_AudioNoiseReduction": "Audio Noise Reduction",
    "JN_AudioAutoTune": "Audio Auto Tune",

    "JN_AudioSampleRate": "Audio Sample Rate",
    "JN_AudioNormalize": "Audio Normalize",
    "JN_AudioSpeed": "Audio Speed",
    "JN_AudioTempo": "Audio Tempo",
    "JN_AudioVolume": "Audio Volume",
    "JN_AudioPitch": "Audio Pitch",
    "JN_AudioReverberation": "Audio Reverberation",
}
