CATEGORY_PACKAGE = "JN"

CATEGORY_AUDIO = f"{CATEGORY_PACKAGE}/Audio"
CATEGORY_AUDIO_CHANNELS = f"{CATEGORY_PACKAGE}/Audio/Channels"
CATEGORY_AUDIO_EDIT = f"{CATEGORY_PACKAGE}/Audio/Edit"
CATEGORY_AUDIO_SAMPLES = f"{CATEGORY_PACKAGE}/Audio/Samples"
CATEGORY_AUDIO_MEOW = f"{CATEGORY_PACKAGE}/Audio/Meow"
CATEGORY_AUDIO_MEOW_TTS = f"{CATEGORY_PACKAGE}/Audio/Meow/TTS"
CATEGORY_AUDIO_MEOW_VC = f"{CATEGORY_PACKAGE}/Audio/Meow/VC"
CATEGORY_IMAGE = f"{CATEGORY_PACKAGE}/Image"
CATEGORY_IMAGE_AREA = f"{CATEGORY_PACKAGE}/Image/Area"
CATEGORY_IMAGE_BLIP = f"{CATEGORY_PACKAGE}/Image/Blip"
CATEGORY_IMAGE_FACE = f"{CATEGORY_PACKAGE}/Image/Face"
CATEGORY_OTHER = f"{CATEGORY_PACKAGE}/Other"
CATEGORY_PATCH = f"{CATEGORY_PACKAGE}/Patch"
CATEGORY_PRIMITIVE = f"{CATEGORY_PACKAGE}/Primitive"
CATEGORY_PRIMITIVE_CONVERSION = f"{CATEGORY_PACKAGE}/Primitive/Conversion"
CATEGORY_PRIMITIVE_PROCESS = f"{CATEGORY_PACKAGE}/Primitive/Process"
CATEGORY_SAMPLING = f"{CATEGORY_PACKAGE}/Sampling"
CATEGORY_WORKFLOW = f"{CATEGORY_PACKAGE}/Workflow"

DIRECTIONS = ["none", "both", "horizontal", "vertical"]

DEVICES = ["gpu", "cpu"]

import comfy
import torch

def get_device(device):
    if device == "gpu":
        device = comfy.model_management.get_torch_device()
    else:
        device = torch.device("cpu")

    return device

