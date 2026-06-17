import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())

from models.resnet_models import get_resnet_model
from models.preprocess import get_transform
from utils.embedding_utils import get_embedding
from utils.faiss_utils import build_index


def register():

    model = get_resnet_model()
    transform = get_transform()

    embeddings = []
    ids = []

    data_path = "data/raw"

    if not os.path.exists(data_path):
        print(" data/raw folder not found.")
        return

    for animal_id in os.listdir(data_path):
        animal_folder = os.path.join(data_path, animal_id)

        if not os.path.isdir(animal_folder):
            continue

        print("Processing:", animal_id)

        for img_name in os.listdir(animal_folder):

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(animal_folder, img_name)

            emb = get_embedding(img_path, model, transform)

            # Convert tensor to numpy if needed
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()

            emb = np.array(emb).reshape(-1).astype("float32")

            # DO NOT NORMALIZE HERE
            # emb = emb / np.linalg.norm(emb)

            embeddings.append(emb)
            ids.append(animal_id)

            # print("Processed:", img_name)

    if len(embeddings) == 0:
        print(" No images found.")
        return

    embeddings = np.array(embeddings).astype("float32")
    ids = np.array(ids)

    os.makedirs("data/embeddings", exist_ok=True)

    np.save("data/embeddings/embeddings.npy", embeddings)
    np.save("data/embeddings/ids.npy", ids)

    print("Embeddings saved.")

    # Rebuild FAISS index
    build_index()
    print("FAISS index rebuilt successfully.")


if __name__ == "__main__":
    register()
    