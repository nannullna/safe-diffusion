from typing import List, Union, Callable, Tuple, Optional
from collections import OrderedDict
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


ALL_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif']


class ImagePathDataset(Dataset):
    def __init__(self, img_files: List[str], transform: Union[Callable, T.Compose]=None):
        self.img_files = img_files
        self.transform = transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_transform(
    size: int=256, 
    normalize: Optional[Union[bool, Tuple[Tuple[float], Tuple[float]]]]=None,
    center_crop: bool=True,
) -> T.Compose:
    transforms = []
    
    if size is not None:
        transforms.append(T.Resize(size, interpolation=T.InterpolationMode.BICUBIC))
        if center_crop:
            transforms.append(T.CenterCrop(size))
    transforms.append(T.ToTensor())

    if isinstance(normalize, bool) and normalize:
        # same as T.Lambda(lambda x: (x - 0.5) * 2) for [-1, 1] normalization
        transforms.append(T.Normalize(0.5, 0.5))
    elif isinstance(normalize, tuple) and len(normalize) == 2:
        # mean, std
        transforms.append(T.Normalize(normalize[0], normalize[1]))
    
    return T.Compose(transforms)


def get_img_files(path: str, exts: Union[str, List[str]]=ALL_EXTS, sort: bool=True) -> List[str]:
    """
    Gets all files in a directory with given extensions.
    Returns a sorted list of files by index if sort is True.
    """
    if isinstance(exts, str):
        exts = [exts]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, f'*{ext}')))
        files.extend(glob(os.path.join(path, f'*{ext.upper()}')))
    if sort:
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return files


def match_files(files1: List[str], files2: List[str]) -> Tuple[List[str], List[str]]:
    """
    Matches files in two lists by number indices. If ignore_ext is True, ignores extension.
    """
    files1_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in files1]
    files2_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in files2]
    
    files1_ids = set(files1_ids)
    files2_ids = set(files2_ids)
    
    common_files = files1_ids.intersection(files2_ids)

    files1 = [f for f in files1 if int(os.path.splitext(os.path.basename(f))[0]) in common_files]
    files2 = [f for f in files2 if int(os.path.splitext(os.path.basename(f))[0]) in common_files]
    
    return files1, files2


def gather_img_tensors(tensors: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor) and tensors.ndim == 3:
        tensors = tensors.unsqueeze(0)
    elif isinstance(tensors, list) and isinstance(tensors[0], torch.Tensor):
        if tensors[0].ndim == 3:
            tensors = torch.stack(tensors, dim=0)
        elif tensors[0].ndim == 4:
            tensors = torch.cat(tensors, dim=0)
    return tensors


def read_prompt_to_ids(path: Optional[str]=None, prompts: Optional[List[str]]=None) -> OrderedDict:
    """Read the prompts txt to get correspoding case_number and prompts.
    prompt.txt should be in the format (each corresponding to a single image):
    ```
    japan body
    japan body
    ...
    japan body
    america body
    ...
    ```

    Returns an OrderedDict mapping each prompt to a list of case numbers as follows:
    ```
    {
        "japan body": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "america body": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ...
    }
    ```
    """
    if prompts is None:
        if path is None:
            raise ValueError('Either prompts or path must be provided.')
        with open(path, 'r') as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [prompt.strip() for prompt in prompts]
    
    prompt_to_ids = OrderedDict()
    for idx, prompt in enumerate(prompts):
        if prompt not in prompt_to_ids:
            prompt_to_ids[prompt] = [idx]
        else:
            prompt_to_ids[prompt].append(idx)
    return prompt_to_ids