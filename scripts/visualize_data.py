
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fedmed.data import DATA_ROOT, list_cases, normalize_channels, labels_to_regions

# Get one MRI case
cases = list_cases(DATA_ROOT)
image_path = cases[0]["image"]
label_path = cases[0]["label"]

# Load MRI and segmentation label
image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
label = nib.load(str(label_path)).get_fdata(dtype=np.float32)

# Normalize the four MRI modalities
image = normalize_channels(image)

# Convert original labels into three tumor-region masks
masks = labels_to_regions(label)

print("Masks shape:", masks.shape)
print("Mask voxel counts (core, whole, enhancing):",
      masks.sum(axis=(1, 2, 3)))

# Select a central axial slice
tumor_slices = np.where(np.any(label > 0, axis=(0, 1)))[0]
slice_idx = int(tumor_slices[len(tumor_slices) // 2])
print("Selected tumor slice:", slice_idx)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]

for i, name in enumerate(modality_names):
    axes[0, i].imshow(image[:, :, slice_idx, i].T, cmap="gray", origin="lower")
    axes[0, i].set_title(name)
    axes[0, i].axis("off")

axes[1, 0].imshow(image[:, :, slice_idx, 0].T, cmap="gray", origin="lower")
axes[1, 0].imshow(
    label[:, :, slice_idx].T,
    cmap="jet",
    alpha=0.45,
    origin="lower",
    vmin=0,
    vmax=3,
)
axes[1, 0].set_title("Original label overlay")
axes[1, 0].axis("off")

region_names = ["Tumor Core", "Whole Tumor", "Enhancing Tumor"]

for i, name in enumerate(region_names):
    axes[1, i + 1].imshow(
        masks[i, :, :, slice_idx].T,
        cmap="gray",
        origin="lower",
    )
    axes[1, i + 1].set_title(name)
    axes[1, i + 1].axis("off")

plt.tight_layout()

output_path = Path("data_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to: {output_path.resolve()}")

plt.show()
