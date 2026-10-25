import numpy as np
import pytest

from fedmed.data import labels_to_regions, normalize_channels, random_crop, split_cases


def test_labels_to_regions():
    label = np.array([[[0, 1], [2, 3]]])
    regions = labels_to_regions(label)
    assert regions.shape == (3, 1, 2, 2)
    assert regions[0].ravel().tolist() == [0, 0, 1, 1]  # tumor core
    assert regions[1].ravel().tolist() == [0, 1, 1, 1]  # whole tumor
    assert regions[2].ravel().tolist() == [0, 0, 0, 1]  # enhancing tumor


def test_normalize_keeps_background_zero():
    rng = np.random.default_rng(0)
    image = rng.uniform(1, 100, size=(4, 8, 8, 8)).astype(np.float32)
    image[:, :2] = 0
    out = normalize_channels(image)
    assert out.shape == image.shape
    assert np.all(out[:, :2] == 0)
    for c in range(4):
        values = out[c][image[c] != 0]
        assert abs(values.mean()) < 1e-4
        assert abs(values.std() - 1) < 1e-3


def test_random_crop_shape_and_tumor_centering():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(4, 20, 20, 20)).astype(np.float32)
    mask = np.zeros((3, 20, 20, 20), dtype=np.float32)
    mask[:, 10, 10, 10] = 1
    image_crop, mask_crop = random_crop(image, mask, (8, 8, 8), rng, foreground_prob=1.0)
    assert image_crop.shape == (4, 8, 8, 8)
    assert mask_crop.shape == (3, 8, 8, 8)
    assert mask_crop.sum() > 0


def test_random_crop_rejects_oversized_crop():
    rng = np.random.default_rng(0)
    image = np.zeros((4, 10, 10, 10), dtype=np.float32)
    mask = np.zeros((3, 10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        random_crop(image, mask, (12, 12, 12), rng)


def test_split_is_disjoint_and_reproducible():
    cases = [{"id": i} for i in range(20)]
    train, val = split_cases(cases, val_fraction=0.25, seed=1)
    assert len(train) == 15 and len(val) == 5
    assert not {c["id"] for c in train} & {c["id"] for c in val}
    _, val_again = split_cases(cases, val_fraction=0.25, seed=1)
    assert val == val_again