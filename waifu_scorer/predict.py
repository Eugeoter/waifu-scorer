import torch
import clip
import os
from PIL import Image
from typing import List
from .utils import get_model_cls

WAIFU_FILTER_V1_MODEL_REPO = 'Eugeoter/waifu-filter-v1/waifu-filter-v1.pth'


def download_from_url(url):
    from huggingface_hub import hf_hub_download
    split = url.split("/")
    username, repo_id, model_name = split[-3], split[-2], split[-1]
    model_path = hf_hub_download(f"{username}/{repo_id}", model_name)
    return model_path


def load_model(model_path: str = None, model_type='mlp', input_size=768, device: str = 'cuda', dtype=torch.float32):
    model_cls = get_model_cls(model_type)
    model = model_cls(input_size=input_size)
    if not os.path.isfile(model_path):
        model_path = download_from_url(model_path)
    s = torch.load(model_path, map_location=device)
    model.load_state_dict(s)
    model.to(device=device, dtype=dtype)
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


class WaifuScorer:
    def __init__(self, model_path: str = WAIFU_FILTER_V1_MODEL_REPO, model_type='mlp', device: str = None, dtype=torch.float32):
        print(f"loading model from `{model_path}`...")
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp = load_model(model_path, model_type=model_type, input_size=768, device=device, dtype=dtype)
        self.mlp.eval()
        self.model2, self.preprocess = clip.load("ViT-L/14", device=device)
        self.device = self.mlp.device
        self.dtype = self.mlp.dtype
        print(f"model loaded: cls={model_type} | device={self.device} | dtype={self.dtype}")

    @torch.no_grad()
    def predict(self, images: List[Image.Image]) -> float:
        images = encode_images(images, self.model2, self.preprocess, device=self.device).to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(images)
        scores = predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()
        return scores
