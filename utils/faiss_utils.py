import os
import numpy as np
import faiss


INDEX_PATH = "data/embeddings/faiss_index.index"
EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"


def build_index():
    """
    Build FAISS index from stored embeddings
    """

    if not os.path.exists(EMBEDDINGS_PATH):
        print("Embeddings file not found.")
        return None

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    if embeddings.ndim != 2:
        print("Embeddings shape invalid:", embeddings.shape)
        return None

    dimension = embeddings.shape[1]

    # ✅ SAFE NORMALIZATION
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10   # prevent division by zero
    embeddings = embeddings / norms

    # Cosine similarity using Inner Product
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("FAISS index built successfully.")
    print("Total vectors in index:", index.ntotal)
    print("Embedding dimension:", dimension)

    return index


def load_index():
    """
    Load FAISS index (if exists), otherwise build new one
    """

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print("FAISS index loaded.")
        print("Total vectors:", index.ntotal)
        return index
    else:
        print("Index not found. Building new index...")
        return build_index()