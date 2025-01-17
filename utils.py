from PIL import Image
from torchvision.transforms import ToPILImage, PILToTensor
import torch

import math


def tensor_to_pil(tensor):
    """Convert a ComfyUI image tensor to PIL Image"""
    to_pil = ToPILImage()
    if len(tensor.shape) == 4:
        # ComfyUI: [B,H,W,C] -> torchvision: [C,H,W]
        return to_pil(tensor[0].permute(2, 0, 1))
    else:
        # MASK [B,H,W] -> torchvision: [C,H,W]
        return to_pil(tensor[0].unsqueeze(-1).permute(2, 0, 1))


def pil_to_tensor(pil_image):
    # Convert PIL image to tensor with proper scaling
    tensor = PILToTensor()(pil_image).float()  # Convert to float
    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    tensor = tensor / 255.0 if tensor.max() > 1.0 else tensor  # Scale to [0,1]
    return tensor.unsqueeze(0)  # Add batch dimension


def numpy_to_tensor(numpy_image, type="image"):
    # (H x W x C). to [[B,H,W,C]
    if type == "image":
        return torch.from_numpy(numpy_image).unsqueeze(0)
    elif type == "mask":
        # (H x W x C) to  [B,H,W]
        return torch.from_numpy(numpy_image).unsqueeze(0)


def resize_image(img, target_size=768):
    width, height = img.size

    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img


def pad_and_resize(
    im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS
):
    old_width, old_height = im.size

    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)

    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new("RGB", (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h


def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size

    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h

    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im
