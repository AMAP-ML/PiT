"""
Simplified dataset loader for ImageNet denoising training.
Only supports CUDA stream-based fast loading.
"""

import os
import glob
import random

import numpy as np
from PIL import Image
from absl import logging

import blobfile as bf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from lib.utils.dist import (
    get_world_size, get_rank,
)

class DataPrefetcher:
    """
    CUDA stream-based data prefetcher for fast GPU transfer.
    Normalizes uint8 images to float on GPU.
    """
    def __init__(self, loader, mean, std):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self._preload()

    def _preload(self):
        # loader will never return None or raise `StopIteration``
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.div_(255).sub_(self.mean).div_(self.std)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.next_input
        labels = self.next_target
        if images is not None:
            images.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
        self._preload()

        return images, {"y": labels}

    def __iter__(self):
        return self


def fast_collate(batch):
    """Collate images as uint8 tensors for GPU-side normalization."""
    if not isinstance(batch[0], (tuple, list)):
        return torch.tensor(batch, dtype=torch.int64)

    imgs = [img[0] for img in batch]
    if batch[0][1] is not None:
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    else:
        targets = None

    width = imgs[0].size[0]
    height = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, height, width), dtype=torch.uint8).contiguous()
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class ImageNetDataset(Dataset):
    """ImageNet dataset with class-conditional CFG dropout."""

    def __init__(
        self,
        image_shape,
        image_paths,
        classes=None,
        cfg_drop_rate=0.0,
        std="imagenet",
        horizontal_flip=True,
        label_only=False,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.local_images = image_paths
        self.local_classes = classes

        if std == "simple":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif std == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229 * 2, 0.224 * 2, 0.225 * 2]

        self.cfg_drop_rate = cfg_drop_rate
        self.num_classes = len(set(self.local_classes)) if self.local_classes is not None else 0

        if classes is not None:
            self.num_labels = max(classes) + 1
            cnt = dict(zip(*np.unique(classes, return_counts=True)))
            self.label_counts = torch.tensor([cnt[k] for k in range(self.num_labels)]).float()
        else:
            self.num_labels = 0
            self.label_counts = None

        self.augmentation = transforms.RandomHorizontalFlip() if horizontal_flip else None
        self.label_only = label_only
        if self.label_only:
            # enable in balanced sampling
            # assume at most 50 samples per class, equivalent to 50K samples in total when measuring FID50k
            max_sample_per_class = 50
            self.local_images = [i for i in range(self.num_labels * max_sample_per_class)]
            self.local_classes = [i for i in range(self.num_labels) for j in range(max_sample_per_class)]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        label = self.local_classes[idx]

        if self.label_only:
            return label

        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.augmentation is not None:
            pil_image = self.augmentation(pil_image)

        # Apply CFG dropout: replace label with num_classes (unconditional token)
        if random.random() < self.cfg_drop_rate:
            label = self.num_classes

        return pil_image, label

    def data_shape(self):
        return self.image_shape

    def unpreprocess(self, x):
        _mean = torch.as_tensor(self.mean)[None, :, None, None].to(x.device)
        _std = torch.as_tensor(self.std)[None, :, None, None].to(x.device)
        return x * _std + _mean

    def sample_label(self, num_samples, device):
        if self.label_counts is not None:
            return {"y": torch.multinomial(self.label_counts, num_samples, replacement=True).to(device)}
        return {}

    def get_uncond_token(self, batch_size, device):
        return {"y": torch.full((batch_size,), self.num_classes, dtype=torch.long, device=device)}


def create_dataloader(
    data_dir,
    batch_size,
    image_shape,
    num_workers=8,
    cfg_drop_rate=0.0,
    std="imagenet",
    horizontal_flip=True,
    label_only=False,
    **kwargs, # Reserve for other dataset params not used here.
):
    """Create ImageNet dataloader with stream-based fast loading."""
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = glob.glob(os.path.join(data_dir, "*/*.png"))
    logging.info(f"total training samples: {len(all_files)}")

    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]

    dataset = ImageNetDataset(
        image_shape, all_files, classes,
        cfg_drop_rate=cfg_drop_rate,
        std=std,
        horizontal_flip=horizontal_flip,
        label_only=label_only,
    )

    dp_world_size = get_world_size()
    dp_rank = get_rank()
    if dp_world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size//dp_world_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=fast_collate,
            sampler=sampler,
        )

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=fast_collate,
        )

    return dataset, loader