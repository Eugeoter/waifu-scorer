import numpy as np
import cv2
import gradio as gr
from PIL import Image
from tqdm import tqdm
from typing import Literal, Callable
from waifu_scorer.predict import WaifuScorer
from waifuset.classes import Dataset, ImageInfo


def parse_interpolation(interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"], type: Literal["pil", "numpy"]):
    if type == 'pil':
        if interpolation == "nearest":
            return Image.NEAREST
        elif interpolation == "bilinear":
            return Image.BILINEAR
        elif interpolation == "bicubic":
            return Image.BICUBIC
        elif interpolation == "lanczos":
            return Image.LANCZOS
        else:
            raise ValueError(f"Unknown interpolation type {interpolation}")
    elif type == 'numpy':
        if interpolation == "nearest":
            return cv2.INTER_NEAREST
        elif interpolation == "bilinear":
            return cv2.INTER_LINEAR
        elif interpolation == "bicubic":
            return cv2.INTER_CUBIC
        elif interpolation == "lanczos":
            return cv2.INTER_LANCZOS4
        else:
            raise ValueError(f"Unknown interpolation type {interpolation}")
    else:
        raise ValueError(f"Unknown image type {type}")


def around_reso(img_w, img_h, reso, divisible=None):
    r"""
    w*h = reso*reso
    w/h = img_w/img_h
    => w = img_ar*h
    => img_ar*h^2 = reso
    => h = sqrt(reso / img_ar)
    """
    reso = reso if isinstance(reso, tuple) else (reso, reso)
    divisible = divisible or 1
    img_ar = img_w / img_h
    around_h = int(np.sqrt(reso[0]*reso[1] / img_ar) // divisible * divisible)
    around_w = int(img_ar * around_h // divisible * divisible)
    return (around_w, around_h)


def around_scale(
    image,
    reso,
    divisible: int = 1,
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "lanczos",
):
    r"""
    Resize the image to the size that is closest to `reso*reso` while maintaining the aspect ratio.
    :param image: The image to resize.
    :param reso: The resolution to resize to. If a tuple is given, it will be used as `(width, height)`.
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        new_w, new_h = around_reso(w, h, reso, divisible=divisible)
        return image.resize((new_w, new_h), resample=parse_interpolation(interpolation, 'pil'))
    elif isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        new_w, new_h = around_reso(w, h, reso, divisible=divisible)
        return cv2.resize(image, (new_w, new_h), interpolation=parse_interpolation(interpolation, 'numpy'))
    else:
        raise TypeError(f"Expected `Image` or `np.ndarray`, got `{type(image)}`")


def gradio_visualize_dataset(
    source,
    image_trans: Callable[[ImageInfo], Image.Image] = None,
    label_func: Callable[[ImageInfo], str] = None,
    max_num_samples: int = None,
    columns: int = 8,
    share: bool = False,
):
    print('launching Gradio demo...')

    dataset = Dataset(source)
    if max_num_samples is not None:
        dataset = dataset.sample(n=max_num_samples, randomly=False)

    image_trans = image_trans or (lambda img_info: Image.open(img_info.image_path))
    label_func = label_func or (lambda img_info: '')

    gallery_images = [(image_trans(img_info), str(label_func(img_info))) for img_info in tqdm(dataset.values(), desc='preprocessing')]

    with gr.Blocks() as demo:
        gallery = gr.Gallery(
            gallery_images,
            columns=columns,
            object_fit='scale-down',
        )

    demo.queue(4)
    demo.launch(share=share)


if __name__ == '__main__':
    dataset = Dataset(
        source=r"D:\AI\datasets\aid\download-2",
        read_attrs=False,
        verbose=True,
    ).sample(n=500, randomly=True)
    predictor_1 = WaifuScorer(
        model_path=r"C:\Users\15070\Desktop\projects\waifu-scorer\models\2024-03-06\3\MLP_best-MSE3.0021_ep729.pth",
        model_type='mlp',
    )
    predictor_2 = WaifuScorer(
        model_path=r"C:\Users\15070\Desktop\projects\waifu-scorer\models\waifu-scorer-v2.pth",
        model_type='mlp',
    )

    batch_size = 4
    res = []

    batches = [dataset.keys()[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    for batch in tqdm(batches, desc='scoring'):
        images = [Image.open(dataset[img_key].image_path) for img_key in batch]
        scores_1 = predictor_1.predict(images)

        for img_key, score_1 in zip(batch, scores_1):
            img_info = dataset[img_key]
            img_info.score_1 = score_1
            res.append(img_info)

        scores_2 = predictor_2.predict(images)
        for img_key, score_2 in zip(batch, scores_2):
            img_info = dataset[img_key]
            img_info.score_2 = score_2

    res = sorted(res, key=lambda img_info: img_info.score_1, reverse=True)
    # ============================= the following code is for visualization, which requires the gradio package =============================

    gradio_visualize_dataset(
        res,
        image_trans=lambda img_info: around_scale(Image.open(img_info.image_path), 512),
        label_func=lambda img_info: f"{img_info.score_1:.2f} / {img_info.score_2:.2f}",
        share=False,
    )
