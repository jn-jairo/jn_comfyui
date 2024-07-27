from functools import reduce
import os
import re
import glob
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import math
import rembg
import hashlib
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms.v2 as T

import folder_paths
import comfy
from comfy.cli_args import args
from comfy_extras.nodes_mask import composite

from ..utils import CATEGORY_IMAGE, CATEGORY_IMAGE_AREA, CATEGORY_IMAGE_BLIP, DIRECTIONS

def resize_background(image, width, height, upscale_method="bilinear", crop="center"):
    samples = image.movedim(-1,1)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    s = s.movedim(1,-1)
    return s

def create_background(batch_size=1, width=1, height=1, color=0, alpha=False, alpha_color=0xFF):
    r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
    g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
    b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)

    if alpha:
        a = torch.full([batch_size, height, width, 1], ((alpha_color) & 0xFF) / 0xFF)
        background = torch.cat((r, g, b, a), dim=-1)
    else:
        background = torch.cat((r, g, b), dim=-1)

    return background

def batch_image(image1, image2):
    if image1.shape[1:] != image2.shape[1:]:
        image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
    s = torch.cat((image1, image2), dim=0)
    return s

def batch_mask(mask1, mask2):
    image1 = mask1.reshape((-1, 1, mask1.shape[-2], mask1.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    image2 = mask2.reshape((-1, 1, mask2.shape[-2], mask2.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    s = batch_image(image1, image2)[:, :, :, 0]
    s = s.reshape((s.shape[0], s.shape[1], s.shape[2]))
    return s

def batch_to_array(image):
    if image is None:
        return [None]

    shape = list(image.shape)
    shape[0] = -1
    image = [image[x].clone().reshape(shape) for x in range(0, image.shape[0])]

    return image

def image_add_mask(image, mask):
    image = image.clone()
    mask = mask.clone().reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1)
    image_masked = torch.cat((image, mask), dim=3)
    return image_masked

def normalize_area(area, width=0xffffffffffffffff, height=0xffffffffffffffff, pad=0, multiple_of=1):
    if not isinstance(area, dict):
        if isinstance(area, list):
            x1 = round(area[0])
            y1 = round(area[1])
            x2 = round(area[2])
            y2 = round(area[3])
        else:
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0

        area = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    multiple_of = max(1, multiple_of)

    area = dict(area)
    area["x1"] = int(max(round((area["x1"] - pad) / multiple_of) * multiple_of, 0))
    area["y1"] = int(max(round((area["y1"] - pad) / multiple_of) * multiple_of, 0))
    area["x2"] = int(min(round((area["x2"] + pad) / multiple_of) * multiple_of, width))
    area["y2"] = int(min(round((area["y2"] + pad) / multiple_of) * multiple_of, height))

    return area

def area_around(a, b, width=0xffffffffffffffff, height=0xffffffffffffffff, pad=0, multiple_of=1):
    a = normalize_area(a)
    b = normalize_area(b)
    return normalize_area({
        "x1": min(a["x1"], b["x1"]),
        "y1": min(a["y1"], b["y1"]),
        "x2": max(a["x2"], b["x2"]),
        "y2": max(a["y2"], b["y2"]),
    }, width=width, height=height, pad=pad, multiple_of=multiple_of)

def get_crop_region(mask, pad=0, multiple_of=1):
    """finds a rectangular region that contains all masked areas in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""

    _, h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, :, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, :, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[:, i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[:, i] == 0).all():
            break
        crop_bottom += 1

    multiple_of = max(1, multiple_of)

    mask_area = {
        "x1": int(max(round((crop_left-pad) / multiple_of) * multiple_of, 0)),
        "y1": int(max(round((crop_top-pad) / multiple_of) * multiple_of, 0)),
        "x2": int(min(round((w - crop_right + pad) / multiple_of) * multiple_of, w)),
        "y2": int(min(round((h - crop_bottom + pad) / multiple_of) * multiple_of, h)),
    }

    if mask_area["x1"] >= mask_area["x2"] or mask_area["y1"] >= mask_area["y2"]:
        mask_area = {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
        }

    return mask_area

def mask_to_area(mask, multiple_of=1):
    return get_crop_region(mask, pad=0, multiple_of=multiple_of)

def area_to_mask(area, width, height, multiple_of=1):
    area = normalize_area(area, width=width, height=height, multiple_of=multiple_of)
    area_width = area["x2"] - area["x1"]
    area_height = area["y2"] - area["y1"]
    area_mask = torch.full((1, area_height, area_width), 1, dtype=torch.float32, device="cpu")
    mask = torch.full((1, height, width), 0, dtype=torch.float32, device="cpu")

    image_area_mask = area_mask.reshape((-1, 1, area_mask.shape[-2], area_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    image_mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    image = composite(destination=image_mask.clone().movedim(-1, 1), source=image_area_mask.clone().movedim(-1, 1), x=area["x1"], y=area["y1"], multiplier=1).movedim(1, -1)

    mask = image[:, :, :, 0].clone()

    return mask

class JN_AreaNormalize:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "area": ("AREA",),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, area, multiple_of=1, image=None, width=0, height=0):
        kwargs = {}

        if image is not None:
            (batches, height, width, channels) = image.shape

        if width > 0:
            kwargs["width"] = width

        if height > 0:
            kwargs["height"] = height

        area = normalize_area(area, multiple_of=multiple_of, **kwargs)

        return (area,)

class JN_AreaAround:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "areas": ("*", {"multiple": True}),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, areas, multiple_of=1, image=None, width=0, height=0):
        areas = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), areas, [None])
        areas = [area for area in areas if area is not None]

        kwargs = {}

        if image is not None:
            (batches, height, width, channels) = image.shape

        if width > 0:
            kwargs["width"] = width

        if height > 0:
            kwargs["height"] = height

        area = normalize_area(reduce(lambda a, b: area_around(a, b, multiple_of=multiple_of, **kwargs), areas), multiple_of=multiple_of, **kwargs)

        return (area,)

class JN_AreaToMask:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("MASK_ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "areas": ("*", {"multiple": True}),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },

        }

    def run(self, areas, multiple_of=1, image=None, width=0, height=0):
        areas = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), areas, [None])
        areas = [area for area in areas if area is not None]

        kwargs = {}

        if image is not None:
            (batches, height, width, channels) = image.shape

        if width > 0:
            kwargs["width"] = width

        if height > 0:
            kwargs["height"] = height


        masks = [area_to_mask(area, multiple_of=multiple_of, **kwargs) for area in areas]

        return (masks,)

class JN_MaskToArea:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("AREA_ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("*", {"multiple": True}),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, masks, multiple_of=1):
        masks = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), masks, [None])
        masks = [mask for mask in masks if mask is not None]

        areas = [mask_to_area(mask, multiple_of=multiple_of) for mask in masks]

        return (areas,)

class JN_AreaXY:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "x2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, x1, y1, x2, y2):
        area = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        return (area,)

class JN_AreaWidthHeight:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, x, y, width, height):
        area = {
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
        }
        return (area,)

class JN_AreaInfo:
    CATEGORY = CATEGORY_IMAGE_AREA
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x1", "y1", "x2", "y2", "width", "height")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "area": ("AREA",),
            },
        }

    def run(self, area):
        area = normalize_area(area)

        x1 = area["x1"]
        y1 = area["y1"]
        x2 = area["x2"]
        y2 = area["y2"]

        width = x2 - x1
        height = y2 - y1

        return (x1, y1, x2, y2, width, height)

class JN_ImageInfo:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "ARRAY")
    RETURN_NAMES = ("width", "height", "channels", "batches", "shape")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    def run(self, image):
        (batches, height, width, channels) = image.shape
        shape = list(image.shape)
        return (width, height, channels, batches, shape)

class JN_ImageSharpness:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("FLOAT", "ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    def run(self, images):
        sharpness_array = []

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("L")
            array = np.asarray(img, dtype=np.int32)

            gy, gx = np.gradient(array)
            gnorm = np.sqrt(gx**2 + gy**2)
            sharpness = np.average(gnorm)

            sharpness_array.append(sharpness)

        sharpness_average = np.average(sharpness_array)

        return (sharpness_average, sharpness_array)

class JN_MaskInfo:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("INT", "INT", "INT", "ARRAY")
    RETURN_NAMES = ("width", "height", "batches", "shape")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    def run(self, mask):
        mask = mask.clone().reshape((-1, mask.shape[-2], mask.shape[-1]))
        (batches, height, width) = mask.shape
        shape = list(mask.shape)
        return (width, height, batches, shape)

class JN_ImageAddMask:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def run(self, image, mask):
        image_masked = image_add_mask(image, mask)
        return (image_masked,)

class JN_ImageSquare:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    METHODS = ["crop", "pad"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (s.METHODS,),
                "images": ("*", {"multiple": True}),
            },
            "optional": {
                "background_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            },
        }

    def run(self, images, method="crop", background_color=0xFFFFFF):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        if method == "pad":
            s = JN_ImageBatch().run(images=images, method="pad", background_color=background_color, aspect_ratio_width=1, aspect_ratio_height=1)[0]
        else:
            for idx, image in enumerate(images):
                image = self.crop_squares(image, background_color=background_color)
                images[idx] = image.clone()

            s = JN_ImageBatch().run(images=images, method="crop", background_color=background_color, aspect_ratio_width=1, aspect_ratio_height=1)[0]

        return (s,)

    def crop_squares(self, image, background_color=0xFFFFFF):
        image = image.clone().movedim(-1,1)
        (batch_size, channels, height, width) = image.shape

        side = min(height, width)
        half_side = math.ceil(side / 2)

        tiles_horizontal = math.ceil(width / side)
        tiles_vertical = math.ceil(height / side)

        diff_width = (side * tiles_horizontal) - width
        diff_height = (side * tiles_vertical) - height

        diff_width_tile = math.ceil(diff_width / max(1, tiles_horizontal - 1))
        diff_height_tile = math.ceil(diff_height / max(1, tiles_vertical - 1))

        total_tiles = (tiles_horizontal + tiles_horizontal - 1) * (tiles_vertical + tiles_vertical - 1)
        total_tiles_batch = total_tiles * batch_size

        output = torch.zeros((total_tiles_batch, channels, side, side))

        b = 0
        for batch in range(batch_size):
            y = 0
            for row in range(0, tiles_vertical):
                x = 0
                for column in range(0, tiles_horizontal):
                    i = column + (row * tiles_horizontal)

                    if x + side > width:
                        x = width - side

                    if y + side > height:
                        y = height - side


                    if tiles_horizontal > 1 and column > 0:
                        nx = x - ((side - diff_width_tile) // 2)
                        output[b] = image[batch, :, y:y+side, nx:nx+side].clone()
                        b += 1

                    if tiles_vertical > 1 and row > 0:
                        ny = y - ((side - diff_height_tile) // 2)
                        output[b] = image[batch, :, ny:ny+side, x:x+side].clone()
                        b += 1

                    output[b] = image[batch, :, y:y+side, x:x+side].clone()
                    b += 1

                    x += side - diff_width_tile
                y += side - diff_height_tile

        output = output.movedim(1,-1)

        return output

class JN_ImageCrop:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "MASK", "AREA")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "area": ("AREA",),
                "mask": ("MASK",),
            },
        }

    def run(self, image, area=None, mask=None, pad=0):
        image = image.clone().movedim(-1,1)
        if mask is not None:
            mask = mask.clone().reshape((-1, mask.shape[-2], mask.shape[-1]))

        if area is not None:
            area = normalize_area(area, image.shape[3], image.shape[2], pad, multiple_of=8)
        else:
            area = get_crop_region(mask, pad, multiple_of=8)

        cropped_image = image[:, :, area["y1"]:area["y2"], area["x1"]:area["x2"]]

        if mask is not None:
            cropped_mask = mask[:, area["y1"]:area["y2"], area["x1"]:area["x2"]]
        else:
            cropped_mask = None

        cropped_image = cropped_image.movedim(1,-1)

        return (cropped_image, cropped_mask, area)

class JN_ImageUncrop:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "area": ("AREA",),
                "mask": ("MASK",),
            }
        }

    def run(self, destination, source, resize_source, area=None, mask=None):
        x = 0
        y = 0
        if area is not None:
            x = area["x1"]
            y = area["y1"]

        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)

        return (output,)

class JN_ImageCenterArea:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "areas": ("*", {"multiple": True}),
                "direction": (DIRECTIONS,),
            },
        }

    def run(self, image, areas, direction="both"):
        if len(areas) == 0:
            return (image,)

        areas = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), areas, [None])
        areas = [area for area in areas if area is not None]
        area = normalize_area(reduce(lambda a, b: area_around(a, b), areas))

        image = self.center(image.clone().movedim(-1, 1), direction, area).movedim(1, -1)

        return (image,)

    def center(self, image, direction, area):
        if direction in ["both", "horizontal"]:
            image = self.center_horizontal(image, area)

        if direction in ["both", "vertical"]:
            image = self.center_vertical(image, area)

        return image

    def center_horizontal(self, image, area):
        (batches, channels, height, width) = image.shape
        area = normalize_area(area, width=width, height=height)

        # Empty area
        if area["x1"] == 0 and area["x2"] == 0 and area["y1"] == 0 and area["y2"] == 0:
            return image

        image = image

        center_area_x = area["x1"] + ((area["x2"] - area["x1"]) // 2)
        center_image_x = width // 2

        tmp = image.clone()

        parts = []

        if center_area_x > center_image_x:
            # 0 -> center_area_x - center_image_x
            parts.append(tmp[:, :, 0:height, 0:(center_area_x - center_image_x)])
            # center_area_x - center_image_x -> center_area_x
            parts.append(tmp[:, :, 0:height, (center_area_x - center_image_x):center_area_x])
            # center_area_x -> width
            parts.append(tmp[:, :, 0:height, center_area_x:width])

            parts = parts[1:] + parts[:1]
        else:
            # 0 -> center_area_x
            parts.append(tmp[:, :, 0:height, 0:center_area_x])
            # center_area_x -> width - (center_image_x - center_area_x)
            parts.append(tmp[:, :, 0:height, center_area_x:(width - (center_image_x - center_area_x))])
            # width - (center_image_x - center_area_x) -> width
            parts.append(tmp[:, :, 0:height, (width - (center_image_x - center_area_x)):width])

            parts = parts[-1:] + parts[:-1]

        paste_x = 0

        for part in parts:
            image[:, :, :, paste_x:(paste_x + part.shape[3])] = part
            paste_x += part.shape[3]

        return image

    def center_vertical(self, image, area):
        area_vertical = normalize_area([area["y1"], area["x1"], area["y2"], area["x2"]])
        return self.center_horizontal(image.movedim(2, 3), area_vertical).movedim(3, 2)

class JN_ImageGrid:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("*", {"multiple": True}),
                "prevent_empty_spots": ("BOOLEAN", {"default": True}),
                "columns": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "rows": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "background_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            },
        }

    def run(self, images, prevent_empty_spots=True, columns=0, rows=0, background_color=0xFFFFFF):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]
        images = reduce(lambda a, b: (a if isinstance(a, list) else batch_to_array(a)) + (b if isinstance(b, list) else batch_to_array(b)), images, [])

        len_images = len(images)

        if len_images == 0:
            empty_image = create_background(width=1, height=1, color=background_color)
            return (empty_image,)

        if len_images == 1:
            return (images[0].clone(),)

        if columns == 0 and rows == 0:
            if prevent_empty_spots:
                rows = math.floor(math.sqrt(len_images))
                while len_images % rows != 0:
                    rows -= 1
            else:
                rows = round(math.sqrt(len_images))

        if columns == 0:
            if rows > len_images:
                rows = len_images
            columns = math.ceil(len_images / rows)
        else:
            if columns > len_images:
                columns = len_images
            rows = math.ceil(len_images / columns)

        columns_width = [0 for i in range(0, columns)]
        rows_height = [0 for i in range(0, rows)]

        for r in range(0, rows):
            for c in range(0, columns):
                i = c + (r * columns)
                if i < len_images:
                    w = images[i].shape[2]
                    h = images[i].shape[1]

                    if w > columns_width[c]:
                        columns_width[c] = w

                    if h > rows_height[r]:
                        rows_height[r] = h

        grid_width = sum(columns_width)
        grid_height = sum(rows_height)

        background = create_background(width=grid_width, height=grid_height, color=background_color).movedim(-1, 1)

        x = 0
        y = 0

        for r in range(0, rows):
            x = 0
            for c in range(0, columns):
                i = c + (r * columns)
                cell_width = columns_width[c]
                cell_height = rows_height[r]
                if i < len_images:
                    image = images[i].clone().movedim(-1, 1)

                    w = image.shape[3]
                    h = image.shape[2]

                    cx = x + (cell_width - w) // 2
                    cy = y + (cell_height - h) // 2

                    background[:, :, cy:(cy+h), cx:(cx+w)] = image

                x += columns_width[c]
            y += rows_height[r]

        background = background.movedim(1, -1)

        return (background,)

class JN_ImageBatch:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    METHODS = ["crop", "pad"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (s.METHODS,),
                "images": ("*", {"multiple": True}),
            },
           "optional": {
                "background_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                "aspect_ratio_width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "aspect_ratio_height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, images, method="crop", background_color=0xFFFFFF, aspect_ratio_width=0, aspect_ratio_height=0):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        if len(images) == 0:
            s = None
        else:
            max_width = None
            max_height = None

            min_width = None
            min_height = None

            min_ratio = None
            min_ratio_deviation = None

            for image in images:
                if max_width is None or image.shape[2] > max_width:
                    max_width = image.shape[2]
                if max_height is None or image.shape[1] > max_height:
                    max_height = image.shape[1]

                if min_width is None or image.shape[2] < min_width:
                    min_width = image.shape[2]
                if min_height is None or image.shape[1] < min_height:
                    min_height = image.shape[1]

                ratio = image.shape[2] / image.shape[1]
                ratio_deviation = min(abs(1 - ratio), abs(1 - (1 / ratio)))

                if min_ratio is None or min_ratio_deviation is None or ratio_deviation < min_ratio_deviation:
                    min_ratio = ratio
                    min_ratio_deviation = ratio_deviation

            if aspect_ratio_width > 0 and aspect_ratio_height > 0:
                min_ratio = aspect_ratio_width / aspect_ratio_height

            max_length = max(max_width, max_height)

            if min_ratio > 1:
                width = max_length
                height = max(1, round((1 / min_ratio) * max_length))
            else:
                width = max(1, round(min_ratio * max_length))
                height = max_length

            if method == "pad":
                if width < max_width:
                    width = max_width
                    height = max(1, round((1 / min_ratio) * max_width))

                if height < max_height:
                    width = max(1, round(min_ratio * max_height))
                    height = max_height

            max_width = width
            max_height = height

            for idx, image in enumerate(images):
                original_width = image.shape[2]
                original_height = image.shape[1]
                width = max(1, round(original_width * max_height / original_height))
                height = max(1, round(original_height * max_width / original_width))
                if method == "pad":
                    if width < max_width:
                        height = max_height
                    else:
                        width = max_width
                    image = comfy.utils.common_upscale(image.clone().movedim(-1, 1), width, height, "bilinear", "disabled").movedim(1, -1)

                    x = (max_width - image.shape[2]) // 2
                    y = (max_height - image.shape[1]) // 2

                    background = create_background(batch_size=image.shape[0], width=max_width, height=max_height, color=background_color)

                    image = composite(destination=background.clone().movedim(-1, 1), source=image.clone().movedim(-1, 1), x=x, y=y, multiplier=1).movedim(1, -1)
                else:
                    image = comfy.utils.common_upscale(image.clone().movedim(-1, 1), max_width, max_height, "bilinear", "center").movedim(1, -1)

                images[idx] = image.clone()

            if len(images) == 1:
                s = images[0]
            else:
                s = reduce(lambda a, b: batch_image(a, b), images)

        return (s,)

class JN_MaskBatch:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"

    METHODS = ["crop", "pad"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (s.METHODS,),
                "masks": ("*", {"multiple": True}),
            },
           "optional": {
                "background_color": ("INT", {"default": 0xFF, "min": 0, "max": 0xFF, "step": 1, "display": "color"}),
                "aspect_ratio_width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "aspect_ratio_height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, masks, method="crop", background_color=0xFFFFFF, aspect_ratio_width=0, aspect_ratio_height=0):
        masks = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), masks, [None])
        masks = [mask for mask in masks if mask is not None]

        if len(masks) == 0:
            s = None
        else:
            images = [mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3) for mask in masks]

            background_color = (background_color << 16) + (background_color << 8) + background_color

            images = JN_ImageBatch().run(images=images, method=method, background_color=background_color, aspect_ratio_width=aspect_ratio_width, aspect_ratio_height=aspect_ratio_height)[0]

            s = images[:, :, :, 0]

        return (s,)

class JN_MaskToImage:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("IMAGE_ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("*", {"multiple": True}),
            },
        }

    def run(self, masks):
        masks = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), masks, [None])
        masks = [mask for mask in masks if mask is not None]

        images = [mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3) for mask in masks]

        return (images,)

class JN_ImageToMask:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("MASK_ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("*", {"multiple": True}),
            },
        }

    def run(self, images):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        masks = [image[:, :, :, 0] for image in images]

        return (masks,)

class JN_SaveImage:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("*", {"multiple": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        return self.save_images(images, filename_prefix, prompt, extra_pnginfo)

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        results = []

        filename_prefix += self.prefix_append

        image_list = images
        batch_number = 0

        for images in image_list:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
                batch_number += 1

        return { "ui": { "images": results } }

class JN_PreviewImage(JN_SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("*", {"multiple": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

class JN_PreviewMask(JN_SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("*", {"multiple": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, masks, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        masks = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), masks, [None])
        masks = [mask for mask in masks if mask is not None]

        images = [mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3) for mask in masks]

        return self.save_images(images, filename_prefix, prompt, extra_pnginfo)

class JN_LoadImageDirectory:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("ARRAY", "ARRAY")
    RETURN_NAMES = ("IMAGE_ARRAY", "MASK_ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        # dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
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
        # files = sorted([os.path.join(directory, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f))])

        if limit > 0:
            files = files[offset:offset+limit]

        images_array = []
        masks_array = []

        for file in files:
            (image, mask) = self.load_image(image=file)
            image = image.reshape((-1, image.shape[-3], image.shape[-2], image.shape[-1]))
            mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

            images_array.append(image)
            masks_array.append(mask)

        return (images_array, masks_array)

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, directory, recursive=True, limit=0, offset=0):
        directory_path = folder_paths.get_annotated_filepath(directory)
        # files = sorted([os.path.join(directory, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f))])

        if limit > 0:
            files = files[offset:offset+limit]

        m = hashlib.sha256()

        for file in files:
            image_path = folder_paths.get_annotated_filepath(file)
            with open(image_path, 'rb') as f:
                m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory, **kwargs):
        if not folder_paths.exists_annotated_filepath(directory):
            return "Invalid directory: {}".format(directory)

        return True

class JN_BlipLoader:
    CATEGORY = CATEGORY_IMAGE_BLIP
    RETURN_TYPES = ("BLIP_MODEL", "BLIP_PROCESSOR")
    FUNCTION = "run"

    MODEL_NAMES = {
        "base": "Salesforce/blip-image-captioning-base",
        "large": "Salesforce/blip-image-captioning-large",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (list(s.MODEL_NAMES.keys()),),
                "precision": (["float32", "float16"],),
                "device": (["cpu", "gpu"],),
            },
        }

    def run(self, model, precision, device):
        model_name = self.MODEL_NAMES[model]
        kargs = {}

        if precision == "float16":
            kargs["torch_dtype"] = torch.float16

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name, **kargs)

        if device == "gpu":
            device = comfy.model_management.get_torch_device()
            model = model.to(device)

        return (model, processor)

class JN_Blip:
    CATEGORY = CATEGORY_IMAGE_BLIP
    RETURN_TYPES = ("STRING", "ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("BLIP_MODEL",),
                "processor": ("BLIP_PROCESSOR",),
                "images": ("IMAGE",),
            },
            "optional": {
                "conditioning": ("STRING", {"default": "", "dynamicPrompts": True}),
                "max_new_tokens": ("INT", {"default": 1000, "min": 0, "max": 0xffffffffffffffff}),
                "skip_special_tokens": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, model, processor, images, conditioning="", max_new_tokens=1000, skip_special_tokens=True):
        images = (images.clone().reshape((-1, images.shape[-3], images.shape[-2], images.shape[-1])) * 255).to(torch.uint8)

        kargs = {
            "images": images,
            "return_tensors": "pt",
        }

        if conditioning != "":
            kargs["text"] = [conditioning for i in range(0, images.shape[0])]

        inputs = processor(**kargs).to(model.device, model.dtype)

        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

        texts = []

        for x in range(0, out.shape[0]):
            text = processor.decode(out[x], skip_special_tokens=skip_special_tokens)
            text = self.clear_text(text)
            texts.append(text)

        first_text = texts[0] if len(texts) > 0 else None

        return (first_text, texts)

    def clear_text(self, text):
        remove_texts = [
            "arafed",
        ]

        spaces = "\s*"

        for remove_text in remove_texts:
            text = re.sub(spaces + remove_text + spaces, "", text, flags=re.IGNORECASE)

        return text

class JN_RemBGSession:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("BACKGROUND_SESSION",)
    FUNCTION = "run"

    MODELS = [
        "u2net",
        "u2netp",
        "u2net_human_seg",
        "u2net_cloth_seg",
        "silueta",
        "isnet-general-use",
        "isnet-anime",
        "sam",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (s.MODELS,),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML", "Tensorrt", "Azure"],),
            },
        }

    def run(self, model, provider="CPU"):
        class Session:
            def __init__(self, model, providers):
                self.session = rembg.new_session(model, providers=[provider+"ExecutionProvider"])
            def process(self, image):
                return rembg.remove(image, session=self.session)
            
        return (Session(model, provider),)

class JN_ImageRemoveBackground:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("ARRAY", "ARRAY", "ARRAY")
    RETURN_NAMES = ("IMAGE_MASK_ARRAY", "IMAGE_ARRAY", "MASK_ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session": ("BACKGROUND_SESSION",),
                "images": ("*", {"multiple": True}),
            },
        }

    def run(self, session, images):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        processed_images_masks = []
        processed_images = []
        processed_masks = []

        for image in images:
            image = image.movedim(-1, 1)
            output = []
            for img in image:
                img = T.ToPILImage()(img)
                img = session.process(img)
                output.append(T.ToTensor()(img))

            output = torch.stack(output, dim=0)
            output = output.movedim(1, -1)

            processed_mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])
            processed_image = output[:, :, :, :3]
            processed_image_mask = image_add_mask(processed_image, processed_mask)

            processed_images.append(processed_image)
            processed_masks.append(processed_mask)
            processed_images_masks.append(processed_image_mask)

        return(processed_images_masks, processed_images, processed_masks)

class JN_ImageAddBackground:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("IMAGE_ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("*", {"multiple": True}),
                "masks": ("*", {"multiple": True}),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "background_color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            },
        }

    def run(self, images, masks, background_image=None, background_color=0):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images, [None])
        images = [image for image in images if image is not None]

        masks = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), masks, [None])
        masks = [mask for mask in masks if mask is not None]

        for idx, image in enumerate(images):
            mask = masks[idx] if len(masks) > idx else torch.ones_like(image[:, :, :, 0])

            if background_image is not None:
                background = resize_background(background_image, image.shape[2], image.shape[1])
            else:
                background = create_background(image.shape[0], image.shape[2], image.shape[1], background_color)

            image = composite(background.movedim(-1, 1), image.movedim(-1, 1), 0, 0, mask, 1, False).movedim(1, -1)

            images[idx] = image.clone()

        return (images,)

NODE_CLASS_MAPPINGS = {
    "JN_ImageInfo": JN_ImageInfo,
    "JN_MaskInfo": JN_MaskInfo,
    "JN_ImageSharpness": JN_ImageSharpness,
    "JN_ImageAddMask": JN_ImageAddMask,
    "JN_ImageSquare": JN_ImageSquare,
    "JN_ImageCrop": JN_ImageCrop,
    "JN_ImageUncrop": JN_ImageUncrop,
    "JN_ImageCenterArea": JN_ImageCenterArea,
    "JN_ImageGrid": JN_ImageGrid,
    "JN_ImageBatch": JN_ImageBatch,
    "JN_MaskBatch": JN_MaskBatch,
    "JN_ImageToMask": JN_ImageToMask,
    "JN_MaskToImage": JN_MaskToImage,
    "JN_SaveImage": JN_SaveImage,
    "JN_PreviewImage": JN_PreviewImage,
    "JN_PreviewMask": JN_PreviewMask,
    "JN_LoadImageDirectory": JN_LoadImageDirectory,
    "JN_AreaNormalize": JN_AreaNormalize,
    "JN_AreaAround": JN_AreaAround,
    "JN_AreaXY": JN_AreaXY,
    "JN_AreaWidthHeight": JN_AreaWidthHeight,
    "JN_AreaInfo": JN_AreaInfo,
    "JN_AreaToMask": JN_AreaToMask,
    "JN_MaskToArea": JN_MaskToArea,
    "JN_BlipLoader": JN_BlipLoader,
    "JN_Blip": JN_Blip,
    "JN_RemBGSession": JN_RemBGSession,
    "JN_ImageRemoveBackground": JN_ImageRemoveBackground,
    "JN_ImageAddBackground": JN_ImageAddBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_ImageInfo": "Image Info",
    "JN_ImageSharpness": "Image Sharpness",
    "JN_MaskInfo": "Mask Info",
    "JN_ImageAddMask": "Image Add Mask",
    "JN_ImageSquare": "Image Square",
    "JN_ImageCrop": "Image Crop",
    "JN_ImageUncrop": "Image Uncrop",
    "JN_ImageCenterArea": "Image Center Area",
    "JN_ImageGrid": "Image Grid",
    "JN_ImageBatch": "Image Batch",
    "JN_MaskBatch": "Mask Batch",
    "JN_ImageToMask": "Image To Mask",
    "JN_MaskToImage": "Mask To Image",
    "JN_SaveImage": "Save Image",
    "JN_PreviewImage": "Preview Image",
    "JN_PreviewMask": "Preview Mask",
    "JN_LoadImageDirectory": "Load Image Directory",
    "JN_AreaNormalize": "Area Normalize",
    "JN_AreaAround": "Area Around",
    "JN_AreaXY": "Area X Y",
    "JN_AreaWidthHeight": "Area Width Height",
    "JN_AreaInfo": "Area Info",
    "JN_AreaToMask": "Area To Mask",
    "JN_MaskToArea": "Mask To Area",
    "JN_BlipLoader": "Blip Loader",
    "JN_Blip": "Blip",
    "JN_RemBGSession": "RemBG Session",
    "JN_ImageRemoveBackground": "Image Remove Background",
    "JN_ImageAddBackground": "Image Add Background",
}
