import os
import torch
import h5py
import math
import random
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Callable, Tuple
from tqdm import tqdm
from PIL import Image
from waifuset.classes import Dataset, ImageInfo
from waifuset.utils import log_utils
from .utils import encode_images, load_clip_models, quality_rating


class LaionImageInfo:
    def __init__(
        self,
        img_path=None,
        im_emb_arr=None,
        rating=None,
        im_emb_arr_flipped=None,
        num_repeats=1,
    ):
        self.img_path = img_path
        self.im_emb_arr = im_emb_arr
        self.rating = rating
        self.im_emb_arr_flipped = im_emb_arr_flipped
        self.num_repeats = num_repeats


class LaionDataset:
    def __init__(
        self,
        source,
        cache_to_disk=True,
        cache_path=None,
        batch_size=1,
        clip_batch_size=4,
        model2=None,
        preprocess=None,
        input_size=768,
        rating_func: Callable = quality_rating,
        repeating_func: Callable = None,
        shuffle=True,
        flip_aug: bool = True,
        device='cuda'
    ):
        if model2 is None or preprocess is None:
            model2, preprocess = load_clip_models(device)  # RN50x64
        if cache_to_disk and cache_path is None:
            raise ValueError("cache_path must be specified when cache_to_disk is True.")
        self.source = source
        self.cache_to_disk = cache_to_disk
        self.cache_path = Path(cache_path)
        self.model2, self.preprocess = model2, preprocess
        self.input_size = input_size
        self.rating_func = rating_func
        self.batch_size = batch_size
        self.encoder_batch_size = clip_batch_size
        self.shuffle = shuffle
        self.flip_aug = flip_aug
        self.device = device

        dataset: Dataset = Dataset(source, verbose=True)

        self.image_data = []

        for img_key, img_info in tqdm(dataset.items(), desc='prepare dataset'):
            img_path = img_info.image_path
            rating = self.rating_func(img_info)
            laion_image_info = LaionImageInfo(
                img_path=img_path,
                rating=rating,
            )
            self.register_image_info(laion_image_info)

        rating_counter = {}
        for laion_img_info in tqdm(self.image_data, desc='calculating num repeats (1/2)'):
            # to list
            rating: torch.Tensor = laion_img_info.rating
            rating_counter.setdefault(rating, 0)
            rating_counter[rating] += 1

        for laion_img_info in tqdm(self.image_data, desc='calculating num repeats (2/2)'):
            benchmark = 30000
            num_repeats = benchmark / rating_counter[laion_img_info.rating]
            prob = num_repeats - math.floor(num_repeats)
            num_repeats = math.floor(num_repeats) if random.random() < prob else math.ceil(num_repeats)
            laion_img_info.num_repeats = max(1, num_repeats)

        self.cache_embs()
        self.batches = self.make_batches()

    def register_image_info(self, image_info: LaionImageInfo):
        self.image_data.append(image_info)

    def cache_embs(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        not_cached = []  # list of (image_info, flipped)
        num_cached = 0

        # load cache
        if self.cache_to_disk:
            pbar = tqdm(total=len(self.image_data), desc='loading cache')

            def load_cached_emb(h5, image_info: LaionImageInfo, flipped=False):
                nonlocal num_cached
                image_key = image_info.img_path.stem
                if flipped:
                    image_key = image_key + '_flipped'
                if image_key in h5:
                    im_emb_arr = torch.from_numpy(f[image_key][:])
                    if im_emb_arr.shape[-1] != self.input_size:
                        raise ValueError(f"Input size mismatched. Except {self.input_size} dim, but got {im_emb_arr.shape[-1]} dim loaded. Please check your cache file.")
                    assert im_emb_arr.device == torch.device('cpu'), "flipped image emb should be on cpu"
                    if flipped:
                        image_info.im_emb_arr_flipped = im_emb_arr
                    else:
                        image_info.im_emb_arr = im_emb_arr
                    num_cached += 1
                else:
                    not_cached.append((image_info, flipped))

            if not is_h5_file(self.cache_path):
                # create cache
                log_utils.info(f"cache file not found, creating new cache file: {self.cache_path}")
                with h5py.File(self.cache_path, 'w') as f:
                    pass
            else:
                log_utils.info(f"loading cache file: {self.cache_path}")
            with h5py.File(self.cache_path, 'r') as f:
                for image_info in self.image_data:
                    load_cached_emb(f, image_info, flipped=False)
                    if self.flip_aug:
                        load_cached_emb(f, image_info, flipped=True)
                    pbar.update()
            pbar.close()
        else:
            not_cached = [(image_info, False) for image_info in self.image_data]
            if self.flip_aug:
                not_cached += [(image_info, True) for image_info in self.image_data]

        # encode not-cached images
        if len(not_cached) == 0:
            log_utils.info("all images are cached.")
        else:
            log_utils.info(f"number of cached instances: {num_cached}")
            log_utils.info(f"number of not cached instances: {len(not_cached)}")

            batches = [not_cached[i:i + self.encoder_batch_size] for i in range(0, len(not_cached), self.encoder_batch_size)]
            pbar = tqdm(total=len(batches), desc='encoding images')

            def cache_batch_embs(h5, batch: List[Tuple[LaionImageInfo, bool]]):
                try:
                    images = [Image.open(image_info.img_path) if not flipped else Image.open(image_info.img_path).transpose(Image.FLIP_LEFT_RIGHT) for image_info, flipped in batch]
                except:
                    log_utils.error(f"Error occurred when loading one of the images: {[image_info.img_path for image_info, flipped in batch]}")
                    raise
                im_emb_arrs = encode_images(images, self.model2, self.preprocess, device=self.device)  # shape: [batch_size, input_size]
                for i, item in enumerate(batch):
                    image_info, flipped = item
                    im_emb_arr = im_emb_arrs[i]
                    shape_size = len(im_emb_arr.shape)
                    if shape_size == 1:
                        im_emb_arr = im_emb_arr.unsqueeze(0)
                    elif shape_size == 3:
                        im_emb_arr = im_emb_arr.squeeze(1)

                    image_key = image_info.img_path.stem
                    assert im_emb_arr.device == torch.device('cpu'), "flipped image emb should be on cpu"
                    if flipped:
                        image_key = image_key + '_flipped'
                        image_info.im_emb_arr_flipped = im_emb_arr
                    else:
                        image_info.im_emb_arr = im_emb_arr

                    if self.cache_to_disk:
                        if image_key in h5:
                            continue
                        h5.create_dataset(image_key, data=im_emb_arr.cpu().numpy())

            try:
                h5 = h5py.File(self.cache_path, 'a') if self.cache_to_disk else None
                for batch in batches:
                    cache_batch_embs(h5, batch)
                    pbar.update()
            finally:
                if h5:
                    h5.close()
            pbar.close()

    def make_batches(self):
        batches = []
        repeated_image_data = []
        for image_info in self.image_data:
            repeated_image_data += [image_info] * image_info.num_repeats
        log_utils.info(f"number of instances (repeated): {len(repeated_image_data)}")
        for i in range(0, len(repeated_image_data), self.batch_size):
            batch = repeated_image_data[i:i + self.batch_size]
            batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __getitem__(self, index):
        batch = self.batches[index]
        im_emb_arrs = []
        ratings = []
        for image_info in batch:
            flip = self.flip_aug and random.random() > 0.5
            if not flip:
                im_emb_arr = image_info.im_emb_arr
            else:
                im_emb_arr = image_info.im_emb_arr_flipped
            rating = image_info.rating

            im_emb_arrs.append(im_emb_arr)
            ratings.append(rating)

        im_emb_arrs = torch.cat(im_emb_arrs, dim=0)
        ratings = torch.tensor(ratings).unsqueeze(-1)
        sample = dict(
            im_emb_arrs=im_emb_arrs,
            ratings=ratings,
        )
        return sample

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    return batch[0]


def get_rating_func(rating_func_type: str):
    if rating_func_type == 'quality':
        from .utils import quality_rating
        rating_func = quality_rating
    else:
        raise ValueError(f"Invalid rating type: {rating_func_type}")
    return rating_func


def prepare_dataloader(
    dataset_source,
    cache_to_disk=True,
    cache_path=None,
    batch_size=1,
    clip_batch_size=4,
    model2=None,
    preprocess=None,
    input_size=768,
    rating_func: Callable = quality_rating,
    shuffle=True,
    flip_aug: bool = True,
    device='cuda',
    persistent_workers=False,
    max_data_loader_n_workers=0,
):
    dataset = LaionDataset(
        dataset_source,
        cache_to_disk=cache_to_disk,
        cache_path=cache_path,
        batch_size=batch_size,
        clip_batch_size=clip_batch_size,
        model2=model2,
        preprocess=preprocess,
        input_size=input_size,
        rating_func=rating_func,
        shuffle=shuffle,
        flip_aug=flip_aug,
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # fix to 1
        shuffle=shuffle,
        num_workers=max_data_loader_n_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return dataset, dataloader


def is_h5_file(cache_path):
    if not cache_path or not h5py.is_hdf5(cache_path):
        return False
    return True


# def make_train_data(
#     dataset_source,
#     rating_func: Callable = quality_rating,
#     batch_size=1,
#     flip_aug: bool = True,
#     device='cuda'
# ):
#     model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
#     dataset = Dataset.from_source(dataset_source, verbose=True)
#     x_train = []
#     y_train = []
#     batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
#     for batch in tqdm(batches, desc='encoding images', smoothing=1):
#         im_emb_arr = encode_images([d.pil_img for d in batch], model2, preprocess, device=device)  # shape: [batch_size, 768]
#         ratings = torch.tensor([rating_func(data) for data in batch]).unsqueeze(-1).to(device)  # shape: [batch_size, 1]
#         x_train.append(im_emb_arr)
#         y_train.append(ratings)
#     x_train = torch.cat(x_train, dim=0)
#     y_train = torch.cat(y_train, dim=0)
#     return x_train, y_train


def prepare_dtype(mixed_precision: str):
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def save_model(model, save_path, epoch=None):
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if epoch is not None:
        save_path = save_path.replace('.pth', f'_ep{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    return save_path
