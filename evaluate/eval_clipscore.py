from typing import List, Optional, Tuple, Union
import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import (
    CLIPProcessor, 
    CLIPTextModel, 
    CLIPVisionModel,
    CLIPModel,
)

from tqdm.auto import tqdm

PRETRAINED_MODEL_NAME_OR_PATH = "openai/clip-vit-large-patch14"


class CLIPDataset(Dataset):
    """
    Dataset for CLIP Score
    """
    def __init__(self, img_files: List[str], captions: List[str], processor: CLIPProcessor=None, prefix: str='A photo depicts '):
        assert len(img_files) == len(captions), f"Number of images {len(img_files)} and captions {len(captions)} must be the same length"
        self.img_files = img_files
        self.captions = captions
        self.processor = processor
        self.prefix = prefix
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.img_files[idx])
        caption = self.prefix + self.captions[idx]
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image["pixel_values"] = image["pixel_values"].squeeze(0)
            caption = self.processor(text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            caption["input_ids"] = caption["input_ids"].squeeze(0)
            caption["attention_mask"] = caption["attention_mask"].squeeze(0)
        return image, caption


class CLIPFeatureExtractor(nn.Module):
    """Reuse CLIP Model to reduce memory footprint."""
    def __init__(self, base_model: Union[CLIPTextModel, CLIPVisionModel], projector: nn.Module):
        super(CLIPFeatureExtractor, self).__init__()
        self.base_model = base_model
        self.projector = projector
        # visual_projection for vision_model
        # text_projection for text_model

    def forward(self, *args, **kwargs):
        outputs = self.base_model(*args, **kwargs)
        pooled_output = outputs[1]
        return self.projector(pooled_output)


def get_model_and_processor(pretrained_model_name_or_path: str=PRETRAINED_MODEL_NAME_OR_PATH) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Get CLIP model and processor
    """
    clip_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
    clip_model = CLIPModel.from_pretrained(pretrained_model_name_or_path)
    return clip_model, clip_processor


@torch.no_grad()
def get_clip_score(model: CLIPModel, images, captions, w: float=2.5, device: Optional[torch.device]=None) -> float:
    """
    Calculate CLIPScore from images and captions
    """

    model.eval()

    pixel_values   = images["pixel_values"].to(device)   if device is not None else images["pixel_values"]
    input_ids      = captions["input_ids"].to(device)      if device is not None else captions["input_ids"]
    attention_mask = captions["attention_mask"].to(device) if device is not None else captions["attention_mask"]

    image_features = model.get_image_features(
        pixel_values=pixel_values,
    ) # (B, D)
    text_features = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ) # (B, D)
    similarity = w * F.cosine_similarity(image_features, text_features, dim=1) # (B, )
    # cosine similarities range from -1 to 1, but normally, we get values from 0 to 0.4.
    # so we multiply by 2.5 to get values from 0 to 1.
    score = similarity.mean().item()
    
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                    prog = 'CLIPScore',
                    description = 'Takes the path to images and prompts and gives CLIPScore')
    parser.add_argument('--img_path', help='path to generated images to be evaluated', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to txt prompts (separated by newline), If not provided, assume img_path contains prompts.txt', type=str, required=False, default=None)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--device', help='device to use', type=str, default='cuda:0')
    parser.add_argument('--pretrained_model_name_or_path', help='pretrained model name or path', type=str, default=PRETRAINED_MODEL_NAME_OR_PATH)
    parser.add_argument('--w', help='weight for cosine similarity', type=float, default=1.0)
    parser.add_argument('--ext', help='extention', type=str, default='png')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    clip_model, clip_processor = get_model_and_processor(args.pretrained_model_name_or_path)
    clip_model.to(device)

    if args.prompts_path is None:
        prompts_path = os.path.join(args.img_path, 'prompts.txt')
    else:
        prompts_path = args.prompts_path
    
    with open(prompts_path, 'r') as f:
        captions = f.readlines()
    captions = [caption.strip() for caption in captions]

    img_files = sorted(glob(os.path.join(args.img_path, f"*.{args.ext.replace('.', '').strip()}")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    dataset = CLIPDataset(img_files, captions, clip_processor, prefix="A photo depicts ")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    score, n = 0.0, 0
    tbar = tqdm(dataloader)
    for images, captions in tbar:
        score += get_clip_score(clip_model, images, captions, args.w, device)
        n += len(images)
        tbar.set_description(f"CLIPScore: {score/n:.4f}")
    score /= n

    print(score)


if __name__ == '__main__':
    main()
