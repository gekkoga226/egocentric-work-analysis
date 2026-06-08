import numpy as np
import torch
import open_clip
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"
_model = None
_preprocess = None
_tokenizer = None


def _get_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        logger.info(f"Loading CLIP model {_MODEL_NAME} ({_PRETRAINED})...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        _model.eval()
    return _model, _preprocess, _tokenizer


def embed_frames(
    frames: list[tuple[float, np.ndarray]],
    batch_size: int = 32,
) -> tuple[list[float], np.ndarray]:
    model, preprocess, _ = _get_model()
    timestamps = []
    all_embeddings = []

    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        images = []
        for ts, bgr in batch:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(rgb)))
            timestamps.append(ts)

        tensor = torch.stack(images)
        with torch.no_grad():
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return timestamps, np.vstack(all_embeddings)


def embed_texts(labels: list[str]) -> np.ndarray:
    model, _, tokenizer = _get_model()
    tokens = tokenizer(labels)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()
