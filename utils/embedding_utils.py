import torch
import numpy as np
from PIL import Image


def get_embedding(image_input, model, transform):
    """
    Accepts:
    - image path (string)
    - PIL Image
    """

    # 🔥 Handle both path and PIL image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        embedding = model(image)
    return embedding.view(-1)
    #return embedding.squeeze()