import os
import torch
import numpy as np
from PIL import Image
from .FitDiT.src.utils_mask import get_mask_location

from .FitDiT.src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from .FitDiT.src.transformer_sd3_garm import (
    SD3Transformer2DModel as SD3Transformer2DModel_Garm,
)
from .FitDiT.src.transformer_sd3_vton import (
    SD3Transformer2DModel as SD3Transformer2DModel_Vton,
)
from .FitDiT.src.pose_guider import PoseGuider
from transformers import CLIPVisionModelWithProjection
from .FitDiT.preprocess.humanparsing.run_parsing import Parsing
from .FitDiT.preprocess.dwpose import DWposeDetector
import math
import random


class FitDiTMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "", "placeholder": "Path to FitDiT model directory"}),
                "model_image": ("IMAGE",),  # Model wearing clothes
                "category": (["Upper-body", "Lower-body", "Dresses"],),
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")  # mask, pose_image, model_image
    RETURN_NAMES = ("mask", "pose_image", "model_image")
    FUNCTION = "generate_mask"
    CATEGORY = "FitDiT"

    def __init__(self):
        self.device = "cuda"
        self.model_root = None
        self.dwprocessor = None
        self.parsing_model = None

    def initialize_models(self, model_dir):
        if self.dwprocessor is not None and model_dir == self.model_root:
            return
        
        self.model_root = model_dir
        self.dwprocessor = DWposeDetector(model_root=model_dir, device=self.device)
        self.parsing_model = Parsing(model_root=model_dir, device=self.device)

    def generate_mask(self, model_dir, model_image, category, offset_top, offset_bottom, offset_left, offset_right):
        self.initialize_models(model_dir)
        model_img = Image.fromarray(model_image)
        
        mask_result, pose_image = self._generate_mask(
            model_img, category, offset_top, offset_bottom, offset_left, offset_right
        )
        
        mask = mask_result["layers"][0]
        pose_image = np.array(pose_image)
        
        return (mask, pose_image, model_image)

    def _generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img_det = resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(
                np.array(vton_img_det)[:, :, ::-1]
            )
            candidate[candidate < 0] = 0
            candidate = candidate[0]

            candidate[:, 0] *= vton_img_det.width
            candidate[:, 1] *= vton_img_det.height

            pose_image = pose_image[:, :, ::-1]  # rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(
                category,
                model_parse,
                candidate,
                model_parse.width,
                model_parse.height,
                offset_top,
                offset_bottom,
                offset_left,
                offset_right,
            )
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            return {
                "background": np.array(vton_img.convert("RGBA")),
                "layers": [
                    np.concatenate(
                        (
                            np.array(mask_gray.convert("RGB")),
                            np.array(mask)[:, :, np.newaxis],
                        ),
                        axis=2,
                    )
                ],
                "composite": np.array(masked_vton_img.convert("RGBA")),
            }, pose_image


class FitDiTTryOnNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "", "placeholder": "Path to FitDiT model directory"}),
                "mask": ("IMAGE",),
                "pose_image": ("IMAGE",),
                "model_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "steps": ("INT", {"default": 20, "min": 15, "max": 30}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "resolution": (["768x1024", "1152x1536", "1536x2048"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FitDiT"

    def __init__(self):
        self.device = "cuda"
        self.generator = None
        self.model_root = None

    def initialize_model(self, model_dir):
        """Initialize the model with the given model directory"""
        if self.generator is not None and model_dir == self.model_root:
            return self.generator

        self.model_root = model_dir
        # Initialize model components
        weight_dtype = torch.bfloat16
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
            os.path.join(model_dir, "transformer_garm"),
            torch_dtype=weight_dtype,
        )
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
            os.path.join(model_dir, "transformer_vton"), torch_dtype=weight_dtype
        )
        pose_guider = PoseGuider(
            conditioning_embedding_channels=1536,
            conditioning_channels=3,
            block_out_channels=(32, 64, 256, 512),
        )
        pose_guider.load_state_dict(
            torch.load(
                os.path.join(model_dir, "pose_guider", "diffusion_pytorch_model.bin")
            )
        )

        # Initialize image encoders
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=weight_dtype
        )
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype
        )

        # Move models to device
        pose_guider.to(device=self.device, dtype=weight_dtype)
        image_encoder_large.to(self.device)
        image_encoder_bigG.to(self.device)

        # Initialize pipeline
        pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            model_dir,
            torch_dtype=weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG,
        )
        pipeline.to(self.device)

        # Initialize auxiliary models
        self.dwprocessor = DWposeDetector(model_root=model_dir, device=self.device)
        self.parsing_model = Parsing(model_root=model_dir, device=self.device)

        self.generator = pipeline
        return pipeline

    def generate(self, model_dir, mask, pose_image, model_image, garment_image, steps, guidance_scale, seed, num_images, resolution):
        self.generator = self.initialize_model(model_dir)
        
        # Convert inputs to PIL Images
        model_img = Image.fromarray(model_image)
        garment_img = Image.fromarray(garment_image)
        mask = Image.fromarray(mask[:,:,3])  # Get alpha channel
        pose_image = Image.fromarray(pose_image)

        # Process images
        new_width, new_height = resolution.split("x")
        new_width, new_height = int(new_width), int(new_height)
        
        model_image_size = model_img.size
        garment_img, _, _ = pad_and_resize(
            garment_img, new_width=new_width, new_height=new_height
        )
        model_img, pad_w, pad_h = pad_and_resize(
            model_img, new_width=new_width, new_height=new_height
        )

        mask = mask.convert("L")
        pose_image = pose_image.convert("L")

        with torch.inference_mode():
            result = self.generator(
                height=new_height,
                width=new_width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garment_img,
                model_image=model_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images,
            ).images

        # Unpad and resize results
        output_images = []
        for img in result:
            img = unpad_and_resize(
                img, pad_w, pad_h, model_image_size[0], model_image_size[1]
            )
            output_images.append(np.array(img))

        return (output_images,)


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


# Add node class to NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FitDiTMask": FitDiTMaskNode,
    "FitDiTTryOn": FitDiTTryOnNode,
}

# Add node display name to NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "FitDiTMask": "FitDiT Mask Generator",
    "FitDiTTryOn": "FitDiT Try-On",
}
