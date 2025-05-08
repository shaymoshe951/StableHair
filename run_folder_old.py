import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline
from tqdm import tqdm

def concatenate_images(image_files, output_file, type="pil"):
    if type == "np":
        image_files = [Image.fromarray(img) for img in image_files]
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

class StableHair:
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float16) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size=512):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)

        # load imgs
        source_image = Image.open(source_image).convert("RGB").resize((size, size))
        id = np.array(source_image)
        reference_image = np.array(Image.open(reference_image).convert("RGB").resize((size, size)))
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image_bald,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return id, sample, source_image_bald, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image


def _get_images_filenames_in_folder(folder_path):
    files = os.listdir(folder_path)
    jpg_files = [os.path.join(folder_path,f) for f in files if (f.endswith('.png') or f.endswith('.jpeg')) and os.path.isfile(os.path.join(folder_path, f))]
    return jpg_files


def run_all():
    folder_path = '/workspace/Projects/Stable-Hair/test_imgs/'
    output_path = '/workspace/Projects/Stable-Hair/output'

    model = StableHair(config="/workspace/Projects/Stable-Hair/configs/hair_transfer.yaml", weight_dtype=torch.float32)


    source_files = _get_images_filenames_in_folder(os.path.join(folder_path, 'ID'))
    ref_files = _get_images_filenames_in_folder(os.path.join(folder_path, 'Ref'))

    os.makedirs(output_path, exist_ok=True)
    for id_image in tqdm(source_files):
        for ref_hair in ref_files:
            # id_image = './test_imgs/ID/0.jpg'
            # ref_hair = './test_imgs/Ref/0.jpg'
            # id_image = os.path.join(folder_path, 'ID', id_image_name)
            # ref_hair = os.path.join(folder_path, 'Ref', ref_hair_name)
            # kwargs = OmegaConf.to_container(model.config.inference_kwargs)
            id, image, source_image_bald, reference_image = model.Hair_Transfer(id_image, ref_hair, random_seed = -1, step = 40, guidance_scale = 1.5, scale = 1, controlnet_conditioning_scale = 1, size = 512)
            nsrc = os.path.basename(id_image).split('.')[0]
            nref = os.path.basename(ref_hair).split('.')[0]
            save_name = nsrc+'_'+nref+'.jpg'
            print(save_name)
            output_file = os.path.join(output_path, save_name)
            # concatenate_images([id, source_image_bald, reference_image, (image*255.).astype(np.uint8)], output_file=output_file, type="np")
            concatenate_images([id, reference_image, (image*255.).astype(np.uint8)], output_file=output_file, type="np")


if __name__ == '__main__':
    run_all()

# # Create a Gradio interface
# image1 = gr.Image(label="id_image")
# image2 = gr.Image(label="ref_hair")
# number0 = gr.Slider(minimum=0.5, maximum=1.5, value=1, label="Converter Scale")
# number1 = gr.Slider(minimum=0.0, maximum=3, value=1.0, label="Hair Encoder Scale")
# number2 = gr.Slider(minimum=1.1, maximum=3.0, value=1.5, label="CFG")
# number3 = gr.Slider(minimum=0.1, maximum=2.0, value=1, label="Latent IdentityNet Scale")
# output1 = gr.Image(type="pil", label="Bald_Result")
# output2 = gr.Image(type="pil", label="Transfer Result")

