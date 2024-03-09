import torch
import clip
from PIL import Image
from typing import List, Union
from . import mlp

QUALITY_TO_RATING = {
    'amazing': 10,
    'best': 8.5,
    'high': 7,
    'normal': 5,
    'low': 2.5,
    'worst': 0,
    'horrible': 0,
}

MODEL_TYPE = {
    'mlp': mlp.MLP,
    'res_mlp': mlp.ResMLP,
}


def quality_rating(img_info):
    quality = (img_info.caption.quality or 'normal') if img_info.caption is not None else 'normal'
    rating = QUALITY_TO_RATING[quality]
    return rating


def get_model_cls(model_type) -> Union[mlp.MLP, None]:
    return MODEL_TYPE.get(model_type, mlp.MLP)


def load_clip_models(name: str = "ViT-L/14", device='cuda'):
    model2, preprocess = clip.load(name, device=device)  # RN50x64
    return model2, preprocess


def load_model(model_path: str = None, model_type=None, input_size=768, batch_norm: bool = True, device: str = 'cuda', dtype=None):
    model_cls = get_model_cls(model_type)
    print(f"Loading model from class `{model_cls}`...")
    model_kwargs = {}
    if model_type in ('large', 'res_large'):
        model_kwargs['batch_norm'] = True
    model = model_cls(input_size, **model_kwargs)
    if model_path:
        try:
            s = torch.load(model_path, map_location=device)
            model.load_state_dict(s)
        except Exception as e:
            print(f"Model type mismatch. Desired model type: `{model_type}` (model class: `{model_cls}`).")
            raise e
        model.to(device)
    if dtype:
        model = model.to(dtype=dtype)
    return model


def normalized(a: torch.Tensor, order=2, dim=-1):
    l2 = a.norm(order, dim, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


@torch.no_grad()
def encode_images(images: List[Image.Image], model2, preprocess, device='cuda') -> torch.Tensor:
    if isinstance(images, Image.Image):
        images = [images]
    image_tensors = [preprocess(img).unsqueeze(0) for img in images]
    image_batch = torch.cat(image_tensors).to(device)
    image_features = model2.encode_image(image_batch)
    im_emb_arr = normalized(image_features).cpu().float()
    return im_emb_arr
