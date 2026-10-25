import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("data/Task01_BrainTumour")


def main():
    meta = json.loads((ROOT / "dataset.json").read_text())
    print("Name            :", meta["name"])
    print("Modalities      :", meta["modality"])
    print("Labels          :", meta["labels"])
    print("Training cases  :", meta["numTraining"])

    case = meta["training"][0]
    image = nib.load(ROOT / case["image"])
    label = nib.load(ROOT / case["label"])

    image_data = image.get_fdata()
    label_data = label.get_fdata()

    print()
    print("Case            :", case["image"])
    print("Image shape     :", image_data.shape)
    print("Label shape     :", label_data.shape)
    print("Image min / max :", image_data.min(), "/", image_data.max())
    print("Label values    :", np.unique(label_data))


if __name__ == "__main__":
    main()