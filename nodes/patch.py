import torch
from typing import Optional
import logging

from ..utils import CATEGORY_PATCH, DIRECTIONS

def _flatten(el):
    flattened = [_flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

class JN_SeamlessBorderCrop:
    CATEGORY = CATEGORY_PATCH
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "params": ("SEAMLESS_PARAMS",),
            },
        }

    def run(self, image, params):
        direction = params["direction"]
        border_percent = params["border_percent"]

        image = self.crop(image.clone().movedim(-1, 1), direction, border_percent).movedim(1, -1)

        return (image,)

    def crop(self, tensor, direction, border_percent):
        (batch_size, channels, height, width) = tensor.shape

        if direction in ["both", "horizontal"]:
            gap = min(round(width * border_percent), width // 4)
            tensor = tensor[:, :, :, gap:-gap]

        if direction in ["both", "vertical"]:
            gap = min(round(height * border_percent), height // 4)
            tensor = tensor[:, :, gap:-gap, :]

        return tensor

class JN_SeamlessBorder:
    CATEGORY = CATEGORY_PATCH
    RETURN_TYPES = ("MODEL", "SEAMLESS_PARAMS")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "direction": (DIRECTIONS,),
                "border_percent": ("FLOAT", {"default": 0.125, "min": 0, "max": 0.25, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    def run(self, model, direction="both", border_percent=0.125, start_percent=0.0, end_percent=1.0):
        params = {
            "direction": direction,
            "border_percent": border_percent,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }

        sigma_start = model.model.model_sampling.percent_to_sigma(start_percent)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_percent)

        model_options = model.model_options

        def apply_seamless(tensor, direction, border_percent):
            (batch_size, channels, height, width) = tensor.shape

            if direction in ["both", "horizontal"]:
                gap = min(round(width * border_percent), width // 4)
                tensor[:, :, :, -gap:] = tensor[:, :, :, gap:(gap * 2)]
                tensor[:, :, :, :gap] = tensor[:, :, :, -(gap * 2):-gap]

            if direction in ["both", "vertical"]:
                gap = min(round(height * border_percent), height // 4)
                tensor[:, :, -gap:, :] = tensor[:, :, gap:(gap * 2), :]
                tensor[:, :, :gap, :] = tensor[:, :, -(gap * 2):-gap, :]

            return tensor

        def unet_wrapper_function(apply_model, options):
            input_x = options["input"]
            timestep_ = options["timestep"]
            c = options["c"]

            sigma = timestep_[0].item()

            if sigma <= sigma_start and sigma >= sigma_end:
                input_x = apply_seamless(input_x, direction, border_percent)

            if "model_function_wrapper" in model_options:
                output = model_options["model_function_wrapper"](apply_model, options)
            else:
                output = apply_model(input_x, timestep_, **c)

            if sigma <= sigma_start and sigma >= sigma_end:
                output = apply_seamless(output, direction, border_percent)

            return output

        m = model.clone()
        m.set_model_unet_function_wrapper(unet_wrapper_function)

        return (m, params)

class JN_Seamless:
    CATEGORY = CATEGORY_PATCH
    RETURN_TYPES = ("MODEL", "VAE", "*")
    RETURN_NAMES = ("MODEL", "VAE", "flow")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "direction": (DIRECTIONS,),
                "min_channels": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "max_channels": ("INT", {"default": 100000, "min": 0, "max": 100000}),
            },
        }

    def run(self, model=None, vae=None, flow=None, dependency=None, direction="both", min_channels=0, max_channels=10000):
        padding_mode = self._direction_to_padding_mode(direction)

        if model:
            self._apply_settings(model.model, padding_mode, min_channels, max_channels)
        if vae:
            self._apply_settings(vae.first_stage_model, padding_mode, min_channels, max_channels)

        return (model, vae, flow)

    def _apply_settings(self, model, padding_mode, min_channels, max_channels):
        layers = _flatten(model)

        channels = []
        count = {}
        count_total = 0

        for layer in [layer for layer in layers if isinstance(layer, torch.nn.Conv2d)]:
            layer_channels = layer.in_channels
            in_range = layer_channels >= min_channels and layer_channels <= max_channels

            layer_padding_mode = padding_mode if in_range else "zeros"

            if layer_channels not in channels:
                channels.append(layer_channels)

            if layer_padding_mode == "circular_horizontal" or layer_padding_mode == "circular_vertical":
                layer.padding_mode = layer_padding_mode

                layer.padding_modeX = "circular" if layer_padding_mode == "circular_horizontal" else "constant"
                layer.padding_modeY = "circular" if layer_padding_mode == "circular_vertical" else "constant"

                layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
                layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])

                def make_bound_method(method, current_layer):
                    def bound_method(self, *args, **kwargs): # Add "self" here
                        return method(current_layer, *args, **kwargs)
                    return bound_method

                bound_method = make_bound_method(self.__replacementConv2DConvForward, layer)
                layer._conv_forward = bound_method.__get__(layer, type(layer))
            else:
                layer.padding_mode = layer_padding_mode if layer_padding_mode else "zeros"
                layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)

            if layer_padding_mode not in count:
                count[layer_padding_mode] = 0

            count[layer_padding_mode] += 1
            count_total += 1

            logging.debug(f"JN_Seamless Conv2d in_channels={layer.in_channels} out_channels={layer.out_channels} padding_mode={layer_padding_mode}")

        logging.info("JN_Seamless")

        channels.sort()

        modes_channels = {}

        for channel in channels:
            if channel >= min_channels and channel <= max_channels:
                mode_channel = padding_mode
            else:
                mode_channel = "zeros"

            if mode_channel not in modes_channels:
                modes_channels[mode_channel] = []

            modes_channels[mode_channel].append(channel)

        logging.info("Channels", channels)

        for key, value in count.items():
            percent = round(value * 100 / count_total, 2)
            mode_channels = modes_channels[key] if key in modes_channels else []
            logging.info("Direction:", self._padding_mode_to_direction(key), f"{value}/{count_total}", f"{percent}%", "Channels:", mode_channels)

        return model

    def __replacementConv2DConvForward(self, layer, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        working = torch.nn.functional.pad(input, layer.paddingX, mode=layer.padding_modeX)
        working = torch.nn.functional.pad(working, layer.paddingY, mode=layer.padding_modeY)
        return torch.nn.functional.conv2d(working, weight, bias, layer.stride, (0, 0), layer.dilation, layer.groups)

    @classmethod
    def _direction_to_padding_mode(s, direction):
        direction_map = {
            "both": "circular",
            "horizontal": "circular_horizontal",
            "vertical": "circular_vertical",
            "none": "zeros",
        }

        return direction_map[direction] if direction in direction_map else "zeros"

    @classmethod
    def _padding_mode_to_direction(s, padding_mode):
        padding_mode_map = {
            "circular": "both",
            "circular_horizontal": "horizontal",
            "circular_vertical": "vertical",
            "zeros": "none",
        }

        return padding_mode_map[padding_mode] if padding_mode in padding_mode_map else "none"

NODE_CLASS_MAPPINGS = {
    "JN_Seamless": JN_Seamless,
    "JN_SeamlessBorder": JN_SeamlessBorder,
    "JN_SeamlessBorderCrop": JN_SeamlessBorderCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_Seamless": "Seamless",
    "JN_SeamlessBorder": "Seamless Border",
    "JN_SeamlessBorderCrop": "Seamless Border Crop",
}
