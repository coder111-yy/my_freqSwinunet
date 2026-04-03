import os
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode='reflect')
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode='nearest')
    return image, label


def apply_elastic_transform(image, label, alpha_scale=0.15, sigma_scale=0.04):
    shape = image.shape[:2]
    alpha = shape[0] * alpha_scale
    sigma = shape[0] * sigma_scale

    random_state = np.random.RandomState(None)
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image_t = np.empty_like(image)
    for i in range(image.shape[2]):
        image_t[:, :, i] = ndimage.map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape)

    label_t = ndimage.map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    return image_t, label_t


def add_speckle_noise(image, variance=0.04):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch) * variance
    noisy_image = image + image * gauss
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    return noisy_image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        if random.random() > 0.5:
            image, label = random_rotate(image, label)

        if random.random() > 0.7:
            image, label = apply_elastic_transform(image, label)

        if random.random() > 0.7:
            image = add_speckle_noise(image, variance=0.04)

        return {'image': image, 'label': label}


_VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _strip_extension(name):
    base, ext = os.path.splitext(name)
    return base if ext else name



def _normalize_record(record, default_split=None):
    """
    Normalize one sample record into (source_split, sample_name_without_ext).

    Supported input forms:
      - 'xxx'              -> (default_split, 'xxx')
      - 'train/xxx'        -> ('train', 'xxx')
      - 'test/xxx.jpg'     -> ('test', 'xxx')
      - ('train', 'xxx')   -> ('train', 'xxx')
    """
    if isinstance(record, (tuple, list)) and len(record) == 2:
        return str(record[0]), _strip_extension(str(record[1]))

    record = str(record).strip().replace('\\', '/')
    if not record:
        raise ValueError('Encountered an empty TN3K sample record.')

    if '/' in record:
        source_split, sample_name = record.split('/', 1)
        return source_split, _strip_extension(sample_name)

    if default_split is None:
        raise ValueError(
            f"Sample record '{record}' has no split prefix. Please pass default_split or use 'train/xxx' format."
        )
    return default_split, _strip_extension(record)



def _resolve_existing_file(base_dir, source_split, sub_dir, sample_name):
    for ext in _VALID_IMAGE_EXTS:
        candidate = os.path.join(base_dir, source_split, sub_dir, sample_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'Cannot find file for sample={sample_name} under {os.path.join(base_dir, source_split, sub_dir)} '
        f'with extensions={_VALID_IMAGE_EXTS}'
    )



def read_split_list(list_dir, split):
    txt_path = os.path.join(list_dir, f'{split}.txt')
    with open(txt_path, 'r', encoding='utf-8') as f:
        items = [line.strip() for line in f.readlines() if line.strip()]
    return items



def load_official_tn3k_pool(list_dir):
    """
    Combine the official train/test lists into a single pool for k-fold CV.
    Returns records in 'train/xxx' or 'test/xxx' format so that the original
    folder structure can stay unchanged.
    """
    train_items = [f'train/{_strip_extension(x)}' for x in read_split_list(list_dir, 'train')]
    test_items = [f'test/{_strip_extension(x)}' for x in read_split_list(list_dir, 'test')]
    return train_items + test_items



def build_kfold_records(all_records, num_folds=5, seed=1234, shuffle=True):
    records = list(all_records)
    if len(records) < num_folds:
        raise ValueError(f'Number of samples ({len(records)}) must be >= num_folds ({num_folds}).')

    indices = np.arange(len(records))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    fold_indices = np.array_split(indices, num_folds)
    folds = []
    for fold_id in range(num_folds):
        val_idx = fold_indices[fold_id].tolist()
        train_idx = np.concatenate([fold_indices[i] for i in range(num_folds) if i != fold_id]).tolist()
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        folds.append((train_records, val_records))
    return folds


class TN3KDataset(Dataset):
    def __init__(self, base_dir, list_dir=None, split=None, transform=None, img_size=384, sample_list=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.img_size = img_size

        if sample_list is not None:
            raw_sample_list = list(sample_list)
        else:
            if list_dir is None or split is None:
                raise ValueError('When sample_list is None, both list_dir and split must be provided.')
            raw_sample_list = read_split_list(list_dir, split)

        self.sample_list = [_normalize_record(item, default_split=split) for item in raw_sample_list]

        self.mean = np.array([0.3915, 0.2582, 0.2558], dtype=np.float32)
        self.std = np.array([0.1656, 0.1430, 0.1631], dtype=np.float32)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        source_split, slice_name = self.sample_list[idx]
        img_path = _resolve_existing_file(self.base_dir, source_split, 'images', slice_name)
        mask_path = _resolve_existing_file(self.base_dir, source_split, 'masks', slice_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(mask_path).convert('L')

        image = image.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        label = np.array(label).astype(np.float32) / 255.0
        label = (label >= 0.5).astype(np.float32)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        image, label = sample['image'], sample['label']

        image = (image - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.int64))

        return {
            'image': image,
            'label': label,
            'case_name': slice_name,
            'source_split': source_split,
        }
