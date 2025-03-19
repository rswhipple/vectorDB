import numpy as np

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
