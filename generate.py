from typing import Dict, List, Any, Optional, Type
import argparse
import os
import sys
import warnings
import math
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel,
    StableDiffusionPipelineSafe, SemanticStableDiffusionPipeline,
    DiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from tqdm.auto import tqdm, trange

# from scripts.commons import load_textual_inversion


MAX_INFER_BATCH_SIZE = int(os.getenv("MAX_INFER_BATCH_SIZE", 4))
PIPELINES: Dict[str, Type[DiffusionPipeline]] = {
    "sd": StableDiffusionPipeline,
    "sld": StableDiffusionPipelineSafe,
    "sega": SemanticStableDiffusionPipeline,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="GenerateV2")

    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True,)
    parser.add_argument('--revision', type=str, required=False,)
    parser.add_argument('--variant', type=str, required=False,)
    parser.add_argument('--unet_path', type=str, required=False, default=None,)
    parser.add_argument('--tokenizer_path', type=str, required=False, default=None,)
    parser.add_argument('--text_encoder_path', type=str, required=False, default=None,)

    parser.add_argument('--pipeline_type', type=str, default="sd", choices=["sd", "sld", "sega"])
    parser.add_argument('--pipeline_config', type=str, default=None)
    parser.add_argument('--safety_concept', type=str, default=None)
    
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--start', type=int, required=False, default=0)
    parser.add_argument('--end', type=int, required=False, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False)

    parser.add_argument('--seed', type=int, required=False,)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=25)

    args = parser.parse_args()
    return args


def validate(
    args: argparse.Namespace,
    prompts: List[str],
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: torch.nn.Module,
    weight_dtype: torch.dtype,
    device: torch.device,
    prefix: str = "",
    seeds: Optional[List[int]]=None,
):

    if seeds is not None:
        # assert args.num_images_per_prompt == 1, "Cannot use seeds with num_images_per_prompt > 1. Please duplicate the prompt with different seeds."
        assert len(seeds) == len(prompts), "The number of seeds must match the number of prompts."

    # Default safety concept defined in SafeStableDiffusionPipeline
    safety_concept: Optional[str] = (
        "an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity,"
        " bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child"
        " abuse, brutality, cruelty"
    )

    # Prepare the pipeline and additional kwargs
    pipeline_cls = PIPELINES[args.pipeline_type]

    pipeline = pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.safety_concept is not None:
        safety_concept = args.safety_concept

    if args.pipeline_type == "sd":
        if args.pipeline_config == "neg":
            pipe_kwargs = {"negative_prompt": [safety_concept]}
        else:
            pipe_kwargs = {}
    elif args.pipeline_type == "sld":
        if args.pipeline_config == "weak":
            pipe_kwargs = SafetyConfig.WEAK
        elif args.pipeline_config == "medium":
            pipe_kwargs = SafetyConfig.MEDIUM
        elif args.pipeline_config == "strong":
            pipe_kwargs = SafetyConfig.STRONG
        elif args.pipeline_config == "max":
            pipe_kwargs = SafetyConfig.MAX
        else:
            pipe_kwargs = {}
    elif args.pipeline_type == "sega":
        pipe_kwargs = {
            "editing_prompt": [safety_concept],
            "reverse_editing_direction": [True],
            "edit_guidance_scale": [5],
        }
    else:
        pipe_kwargs = {}

    print(pipeline)
    print(pipe_kwargs)

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # Duplicate prompts and seeds if num_images_per_prompt > 1
    _prompts = [(i, p) for i, p in enumerate(prompts) for _ in range(args.num_images_per_prompt)]
    prompts = [p[1] for p in _prompts]
    indices = [p[0] for p in _prompts]

    if args.image_dir is not None:
        os.makedirs(args.image_dir, exist_ok=True)

    all_image_files, all_prompts = [], []
    num_total_iters = math.ceil(len(prompts) / MAX_INFER_BATCH_SIZE)
    tbar = trange(len(prompts), desc=f"Prompt: {prompts[0]}")
    index = 0
    prev_idx = -1

    for i in range(num_total_iters):

        start_idx = i * MAX_INFER_BATCH_SIZE
        end_idx = min((i + 1) * MAX_INFER_BATCH_SIZE, len(prompts))

        # Manually generate latent codes for random seeds
        curr_prompts, all_latents = [], []
        for j in range(start_idx, end_idx):
            example_idx = indices[j]

            # To continue generating images from a previous run, skip the prompts.
            if not args.overwrite and os.path.exists(os.path.join(args.image_dir, f"{index:06d}.png")):
                index += 1
                tbar.update(1)
                continue
            elif example_idx < args.start:
                index += 1
                tbar.update(1)
                continue
            elif args.end is not None and example_idx >= args.end:
                break
            
            curr_prompts.append(prompts[j])
            if seeds is not None and example_idx != prev_idx:
                generator.manual_seed(seeds[example_idx])
            prev_idx = example_idx
            
            all_latents.append(torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=weight_dtype))

        if len(curr_prompts) == 0:
            continue
            
        all_latents = torch.cat(all_latents, dim=0)
    
        # Prepare to generate images
        tbar.set_description(f"Prompt ({prev_idx}): {curr_prompts[0]}")
        if "negative_prompt" in pipe_kwargs:
            negative_prompt = pipe_kwargs["negative_prompt"][0]
            pipe_kwargs["negative_prompt"] = [negative_prompt] * len(curr_prompts)
        
        images = pipeline(
            prompt=curr_prompts,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            **pipe_kwargs,
        ).images
        
        all_prompts.extend(curr_prompts)
        if args.image_dir is not None:
            for image in images:
                image_file_name = os.path.join(args.image_dir, f"{index:06d}.png")
                image.save(image_file_name)
                all_image_files.append(image_file_name)
                index += 1
        tbar.update(len(curr_prompts))
    
    if args.image_dir is not None:
        with open(os.path.join(args.image_dir, "prompts.txt"), "w") as f:
            for prompt in all_prompts:
                f.write(prompt + "\n")

    del pipeline
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def load_prompts(path: str) -> List[str]:
    if path.endswith('.txt'):
        with open(path, 'r') as f:
            prompts = f.readlines()
            seeds = None
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            prompt_json = json.load(f)
        # prompt_json can be
        # 1. a list of strings, 
        # 2. a list of dictionaries with keys "prompt" and "seed (or evaluation_seed)", 
        # 3. or a dictionary with keys "prompts" and "seeds (or evaluation_seeds)".
        if isinstance(prompt_json, list):
            if isinstance(prompt_json[0], str):
                prompts = prompt_json
                seeds = None
            elif isinstance(prompt_json[0], dict):
                seed_column = 'seed' if 'seed' in prompt_json[0].keys() else 'evaluation_seed'
                prompts = [prompt['prompt'] for prompt in prompt_json]
                seeds = [prompt[seed_column] for prompt in prompt_json]
            else:
                raise ValueError("prompt_json must be a list of strings or a list of dictionaries with keys 'prompt' and 'seed'.")
        elif isinstance(prompt_json, dict):
            prompts = prompt_json['prompts']
            seeds = prompt_json['seeds'] if 'seeds' in prompt_json.keys() else prompt_json['evaluation_seeds']
        
    elif path.endswith('.csv'):
        prompt_df = pd.read_csv(path)
        prompts = prompt_df['prompt'].tolist()
        if "evaluation_seed" in prompt_df.columns:
            seeds = prompt_df['evaluation_seed'].tolist()
        elif "seed" in prompt_df.columns:
            seeds = prompt_df['seed'].tolist()
        else:
            seeds = None
    else:
        raise ValueError("prompt_path must be a .txt, .json, or .csv file.")

    if seeds is not None:
        assert len(seeds) == len(prompts), "The number of seeds must match the number of prompts."
        seeds = [int(seed) for seed in seeds]
    prompts = [prompt.strip() for prompt in prompts]

    return prompts, seeds


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load Models
    if args.unet_path is not None:
        print(f"UNet path is specified. Loading UNet from {args.unet_path}")
        unet = UNet2DConditionModel.from_pretrained(args.unet_path)
    else:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet')
    
    if args.tokenizer_path is not None:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    
    if args.text_encoder_path is not None:
        text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)
    else:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')

    # Load Prompts
    prompts, seeds = load_prompts(args.prompt_path)    

    if args.use_fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # Validate
    validate(
        args,
        prompts=prompts,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        weight_dtype=weight_dtype,
        device=device,
        seeds=seeds,
    )
    


if __name__ == "__main__":
    main()