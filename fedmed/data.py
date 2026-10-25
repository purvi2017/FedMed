
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# Path to the BrainTumour dataset
DATA_ROOT = Path("data/Task01_BrainTumour")

# Index of the whole tumor mask
WHOLE_TUMOR = 1


def list_cases(root=DATA_ROOT):
    # Get the image and label paths from the dataset file
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text())

    return [
        {"image": root / item["image"], "label": root / item["label"]}
        for item in meta["training"]
    ]


def split_cases(cases, val_fraction=0.2, seed=42):
    # Split the cases into training and validation sets
    # A fixed seed keeps the split the same every time
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    order = np.random.default_rng(seed).permutation(len(cases))
    n_val = int(round(len(cases) * val_fraction))
    val_indices = set(order[:n_val].tolist())

    train = [c for i, c in enumerate(cases) if i not in val_indices]
    val = [c for i, c in enumerate(cases) if i in val_indices]

    return train, val


def normalize_channels(image):
    # Normalize each MRI channel using its non-zero voxels
    # Keep background voxels at zero
    out = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[0]):
        channel = image[c]
        mask = channel != 0

        if mask.any():
            mean = channel[mask].mean()
            std = channel[mask].std()
            out[c][mask] = (channel[mask] - mean) / (std + 1e-8)

    return out


def labels_to_regions(label):
    # Create three masks: tumor core, whole tumor and enhancing tumor
    tumor_core = (label == 2) | (label == 3)
    whole_tumor = tumor_core | (label == 1)
    enhancing = label == 3

    return np.stack([tumor_core, whole_tumor, enhancing]).astype(np.float32)


def random_crop(image, mask, crop_size, rng, foreground_prob=0.0):
    # Take a smaller 3D section from the MRI image and its mask
    spatial = image.shape[1:]

    if any(c > s for c, s in zip(crop_size, spatial)):
        raise ValueError("Crop size is larger than the volume.")

    # Sometimes place the crop around a tumor voxel
    # Otherwise, choose a random location
    if (
        foreground_prob > 0
        and rng.random() < foreground_prob
        and mask[WHOLE_TUMOR].any()
    ):
        voxels = np.argwhere(mask[WHOLE_TUMOR] > 0)
        center = voxels[rng.integers(len(voxels))]

        starts = [
            int(np.clip(ctr - c // 2, 0, s - c))
            for ctr, c, s in zip(center, crop_size, spatial)
        ]
    else:
        starts = [
            int(rng.integers(0, s - c + 1))
            for c, s in zip(crop_size, spatial)
        ]

    # Use the same crop location for the image and its mask
    window = tuple(slice(st, st + c) for st, c in zip(starts, crop_size))

    return image[(slice(None),) + window], mask[(slice(None),) + window]


class BrainTumourDataset(Dataset):
    # Load one MRI case whenever it is needed
    # Set crop_size to None to return the full volume
    # Use num_workers=0 for now

    def __init__(self, cases, crop_size=(96, 96, 96), foreground_prob=0.7, seed=0):
        self.cases = cases
        self.crop_size = crop_size
        self.foreground_prob = foreground_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        # Return the total number of cases
        return len(self.cases)

    def __getitem__(self, index):
        # Load the MRI image and its segmentation label
        case = self.cases[index]

        image = np.asarray(nib.load(case["image"]).dataobj, dtype=np.float32)
        label = np.asarray(nib.load(case["label"]).dataobj).astype(np.int16)

        # Move the four MRI channels to the first dimension
        image = normalize_channels(image.transpose(3, 0, 1, 2))

        # Convert the labels into three tumor region masks
        mask = labels_to_regions(label)

        # Use a smaller crop to reduce memory usage
        if self.crop_size is not None:
            image, mask = random_crop(
                image, mask, self.crop_size, self.rng, self.foreground_prob
            )

        # Convert the arrays into PyTorch tensors
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
        )


if __name__ == "__main__":
    # Split the dataset into training and validation cases
    train_cases, val_cases = split_cases(list_cases())

    print("Train cases     :", len(train_cases))
    print("Validation cases:", len(val_cases))

    # Load one training sample and check its shape
    dataset = BrainTumourDataset(train_cases)
    image, mask = dataset[0]

    print("Image crop      :", tuple(image.shape), image.dtype)
    print("Mask crop       :", tuple(mask.shape), mask.dtype)
    print(
        "Mask voxels per region (core, whole, enhancing):",
        mask.sum(dim=(1, 2, 3)).tolist(),
    )
