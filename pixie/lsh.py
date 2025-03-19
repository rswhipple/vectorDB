import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

class LSH:
    """
    A simple Locality Sensitive Hashing implementation for vector search.
    """
    
    def __init__(self, hash_size=8, seed=42):
        """
        Initialize the LSH.
        
        Args:
            hash_size: Number of bits in the hash
            seed: Random seed
        """
        self.hash_size = hash_size
        self.random_state = np.random.RandomState(seed)
        self.hash_tables = {}  # hash -> [doc_ids]
        self.projection_vectors = None
        
    def _generate_projection_vectors(self, dim):
        """
        Generate random projection vectors.
        """
        self.projection_vectors = self.random_state.randn(self.hash_size, dim)
        
    def _hash_vector(self, vector):
        """
        Hash a vector using random projections.
        """
        if self.projection_vectors is None:
            self._generate_projection_vectors(len(vector))
            
        # Project the vector onto random directions
        projections = np.dot(self.projection_vectors, vector)
        
        # Convert to binary hash
        hash_bits = (projections >= 0).astype(int)
        
        # Convert binary array to integer
        hash_value = 0
        for bit in hash_bits:
            hash_value = (hash_value << 1) | bit
            
        return hash_value
    
    def add(self, doc_id: str, vector: np.ndarray):
        """
        Add a vector to the LSH.
        
        Args:
            doc_id: Document ID
            vector: The vector to add
        """
        hash_value = self._hash_vector(vector)
        
        if hash_value not in self.hash_tables:
            self.hash_tables[hash_value] = []
            
        self.hash_tables[hash_value].append(doc_id)
        
    def query(self, vector: np.ndarray) -> List[str]:
        """
        Find documents with the same hash as the query vector.
        
        Args:
            vector: The query vector
            
        Returns:
            List of document IDs
        """
        hash_value = self._hash_vector(vector)
        
        return self.hash_tables.get(hash_value, [])
