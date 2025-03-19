import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import uuid
from .kd_tree import KDTree
from .lsh import LSH

class Pixie:
    """
    A simple vector database that combines KD-Trees and LSH for efficient similarity search.
    """
    
    def __init__(self, embedder, lsh_hash_size=8, n_trees=1):
        """
        Initialize the vector store.
        
        Args:
            embedder: A model that converts text to embeddings
            lsh_hash_size: Number of bits in the LSH hash
            n_trees: Number of KD-Trees to use
        """
        self.embedder = embedder
        self.vectors = {}  # id -> vector
        self.documents = {}  # id -> document
        self.lsh = LSH(hash_size=lsh_hash_size)
        self.trees = [KDTree() for _ in range(n_trees)]
        
    def add(self, doc: str, doc_id: Optional[str] = None) -> str:
        """
        Add a document to the vector store.
        
        Args:
            doc: The document text
            doc_id: Optional document ID
            
        Returns:
            The document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        # Generate embedding
        vector = self.embedder.encode(doc)
        
        # Store document and vector
        self.vectors[doc_id] = vector
        self.documents[doc_id] = doc
        
        # Add to LSH
        self.lsh.add(doc_id, vector)
        
        # Add to KD-Trees
        for tree in self.trees:
            tree.add(doc_id, vector)
            
        return doc_id
    
    def from_docs(self, docs: List[str]) -> List[str]:
        """
        Add multiple documents to the vector store.
        
        Args:
            docs: List of document texts
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc in docs:
            doc_id = self.add(doc)
            doc_ids.append(doc_id)
        return doc_ids
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Find the most similar documents to the query.
        
        Args:
            query: The query text
            top_k: Number of results to return
            
        Returns:
            List of document texts
        """
        # Generate query embedding
        query_vector = self.embedder.encode(query)
        
        # Get candidate IDs from LSH
        lsh_candidates = self.lsh.query(query_vector)
        
        # Get candidate IDs from KD-Trees
        kd_candidates = set()
        for tree in self.trees:
            kd_candidates.update(tree.search(query_vector, k=top_k))
        
        # Combine candidates
        candidates = list(set(lsh_candidates) | kd_candidates)
        
        # If we don't have enough candidates, use all documents
        if len(candidates) < top_k:
            candidates = list(self.vectors.keys())
        
        # Calculate distances
        distances = []
        for doc_id in candidates:
            vector = self.vectors[doc_id]
            distance = np.linalg.norm(query_vector - vector)
            distances.append((doc_id, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return top_k documents
        result = []
        for doc_id, _ in distances[:top_k]:
            result.append(self.documents[doc_id])
            
        return result
