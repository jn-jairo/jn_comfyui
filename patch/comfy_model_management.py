from comfy.model_management import *
import comfy.model_management
import comfy.utils
import logging

def patchComfyModelManagement(config):
    def get_torch_device():
        global directml_enabled
        global cpu_state

        extension_stack = comfy.utils.get_extension_calling()
        if extension_stack is not None:
            for extension in extension_stack:
                if extension in config["extension_device"] and config["extension_device"][extension] is not None:
                    logging.info(f"Device for '{extension }': " + config["extension_device"][extension])
                    return torch.device(config["extension_device"][extension])

        if directml_enabled:
            global directml_device
            return directml_device
        if cpu_state == CPUState.MPS:
            return torch.device("mps")
        if cpu_state == CPUState.CPU:
            return torch.device("cpu")
        else:
            if is_intel_xpu():
                return torch.device("xpu", torch.xpu.current_device())
            else:
                return torch.device(torch.cuda.current_device())

    comfy.model_management.get_torch_device = get_torch_device

PATCHES = {
    "30_comfy_model_management": patchComfyModelManagement,
}
