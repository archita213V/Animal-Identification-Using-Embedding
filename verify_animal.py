import os
import torch
import numpy as np
from PIL import Image
import cv2

from models.sam_detector import SAMDetector
from models.resnet_models import get_resnet_model
from models.preprocess import get_transform
from utils.embedding_utils import get_embedding
from utils.faiss_utils import load_index
import time

THRESHOLD = 0.7


def verify(image_path, model, transform, detector, index, ids):
   
    start = time.time()
    crops = detector.detect_animals(image_path)
    print("Detection time:", time.time()-start)
    detected_ids = set()

    crops = detector.detect_animals(image_path, save_crops=True)

    if len(crops) == 0:
        print("❌ No animals detected")
        return
    found = False


    for crop in crops:
        crop = cv2.resize(crop,(224,224))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)

        query_emb = get_embedding(crop, model, transform)

        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.detach().cpu().numpy()

        query_emb = query_emb.astype("float32").reshape(1, -1)

        norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        query_emb = query_emb / norm

        distances, indices = index.search(query_emb, 1)

        similarity = distances[0][0]
        matched_id = ids[indices[0][0]]

        if similarity > THRESHOLD and matched_id not in detected_ids:

            print("\nMatched Animal:", matched_id)
            print("Similarity:", similarity)
            print("✅ Animal Verified")

            detected_ids.add(matched_id)
            found = True

    if not found:
        print("Animal Not Verified")
if __name__ == "__main__":

    model = get_resnet_model()
    transform = get_transform()
    detector = SAMDetector()

    embeddings_path = "data/embeddings/embeddings.npy"
    ids_path = "data/embeddings/ids.npy"

    if not os.path.exists(embeddings_path) or not os.path.exists(ids_path):
        print("Embeddings file not found. Run registration first.")
        exit()

    ids = np.load(ids_path)
    index = load_index()

    if index is None or index.ntotal == 0:
        print("FAISS index empty.")
        exit()

    test_folder = "data/test_animal"

    for img_name in os.listdir(test_folder):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(test_folder, img_name)
        img_path = img_path.replace("\\", "/")
        print("Testing Image:", img_name)
        verify(img_path, model, transform, detector, index, ids)
        
        
        