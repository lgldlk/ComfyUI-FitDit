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
from .utils import (
    pil_to_tensor,
    tensor_to_pil,
    pad_and_resize,
    unpad_and_resize,
    resize_image,
    numpy_to_tensor,
)


class FitDiTModelLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": (
                    "STRING",
                    {"default": "", "placeholder": "Path to FitDiT model directory"},
                ),
                "use_fp16": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FITDIT_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "FitDiT"

    def __init__(self):
        self.device = "cuda"
        self.model_root = None
        self.dwprocessor = None
        self.parsing_model = None
        self.generator = None

    def load_model(self, model_dir, use_fp16):
        if self.generator is not None and model_dir == self.model_root:
            return (
                {
                    "model_dir": model_dir,
                    "dwprocessor": self.dwprocessor,
                    "parsing_model": self.parsing_model,
                    "generator": self.generator,
                },
            )

        self.model_root = model_dir
        # Initialize DWpose and Parsing models
        self.dwprocessor = DWposeDetector(model_root=model_dir, device=self.device)
        self.parsing_model = Parsing(model_root=model_dir, device=self.device)

        # Initialize main model components
        weight_dtype = torch.float16 if use_fp16 else torch.bfloat16
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
            "openai/clip-vit-large-patch14",
            torch_dtype=weight_dtype,
        )
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            torch_dtype=weight_dtype,
        )

        # Move models to device
        pose_guider.to(device=self.device, dtype=weight_dtype)
        image_encoder_large.to(self.device)
        image_encoder_bigG.to(self.device)

        # Initialize pipeline
        self.generator = StableDiffusion3TryOnPipeline.from_pretrained(
            model_dir,
            torch_dtype=weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG,
        )
        self.generator.to(self.device)

        return (
            {
                "model_dir": model_dir,
                "dwprocessor": self.dwprocessor,
                "parsing_model": self.parsing_model,
                "generator": self.generator,
            },
        )


class FitDiTMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fitdit_model": ("FITDIT_MODEL",),
                "model_image": ("IMAGE",),  # Model wearing clothes
                "category": (["Upper-body", "Lower-body", "Dresses"],),
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")  # mask, pose_image, model_image
    RETURN_NAMES = ("mask", "pose_image", "model_image")
    FUNCTION = "generate_mask"
    CATEGORY = "FitDiT"

    def __init__(self):
        self.device = "cuda"
        self.model_root = None
        self.dwprocessor = None
        self.parsing_model = None

    def generate_mask(
        self,
        fitdit_model,
        model_image,
        category,
        offset_top,
        offset_bottom,
        offset_left,
        offset_right,
    ):
        self.dwprocessor = fitdit_model["dwprocessor"]
        self.parsing_model = fitdit_model["parsing_model"]
        model_img = tensor_to_pil(model_image)

        mask_result, pose_image = self._generate_mask(
            model_img, category, offset_top, offset_bottom, offset_left, offset_right
        )

        mask = mask_result["layers"][0][:, :, 3]
        pose_image = np.array(pose_image)

        return (
            numpy_to_tensor(mask, "mask"),
            numpy_to_tensor(pose_image),
            model_image,
        )

    def _generate_mask(
        self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right
    ):
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
                "fitdit_model": ("FITDIT_MODEL",),
                "mask": ("MASK",),
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

    def generate(
        self,
        fitdit_model,
        mask,
        pose_image,
        model_image,
        garment_image,
        steps,
        guidance_scale,
        seed,
        num_images,
        resolution,
    ):
        self.generator = fitdit_model["generator"]
        # Convert inputs to PIL Images
        model_img = tensor_to_pil(model_image)
        garment_img = tensor_to_pil(garment_image)
        mask = tensor_to_pil(mask)
        pose_image = tensor_to_pil(pose_image)

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
        mask, _, _ = pad_and_resize(
            mask, new_width=new_width, new_height=new_height, pad_color=(0, 0, 0)
        )
        pose_image, _, _ = pad_and_resize(
            pose_image, new_width=new_width, new_height=new_height, pad_color=(0, 0, 0)
        )

        mask = mask.convert("L")

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
            tensor_img = pil_to_tensor((img.convert("RGB")))
            output_images.append(tensor_img)
        res = torch.cat(output_images, dim=0)
        return (res,)


# Add node class to NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FitDiTModelLoader": FitDiTModelLoaderNode,
    "FitDiTMask": FitDiTMaskNode,
    "FitDiTTryOn": FitDiTTryOnNode,
}

# Add node display name to NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "FitDiTModelLoader": "FitDiT Model Loader",
    "FitDiTMask": "FitDiT Mask Generator",
    "FitDiTTryOn": "FitDiT Try-On",
}
