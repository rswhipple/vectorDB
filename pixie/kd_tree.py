import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

class KDNode:
    def __init__(self, doc_id, vector, axis=0, left=None, right=None):
        self.doc_id = doc_id
        self.vector = vector
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    """
    A simple KD-Tree implementation for vector search.
    """
    
    def __init__(self):
        """
        Initialize an empty KD-Tree.
        """
        self.root = None
        self.dimension = None
        
    def add(self, doc_id: str, vector: np.ndarray):
        """
        Add a vector to the KD-Tree.
        
        Args:
            doc_id: Document ID
            vector: The vector to add
        """
        if self.dimension is None:
            self.dimension = len(vector)
        
        self.root = self._insert(self.root, doc_id, vector, 0)
        
    def _insert(self, node, doc_id, vector, depth):
        """
        Recursively insert a vector into the KD-Tree.
        """
        if node is None:
            return KDNode(doc_id, vector, depth % self.dimension)
        
        axis = depth % self.dimension
        
        if vector[axis] < node.vector[axis]:
            node.left = self._insert(node.left, doc_id, vector, depth + 1)
        else:
            node.right = self._insert(node.right, doc_id, vector, depth + 1)
            
        return node
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[str]:
        """
        Find the k nearest neighbors to the query vector.
        
        Args:
            query_vector: The query vector
            k: Number of neighbors to find
            
        Returns:
            List of document IDs
        """
        if self.root is None:
            return []
        
        # Use a priority queue to keep track of the k nearest neighbors
        import heapq
        nearest = []
        
        def _search(node, depth):
            if node is None:
                return
            
            # Calculate distance
            dist = np.linalg.norm(query_vector - node.vector)
            
            # Add to priority queue
            if len(nearest) < k:
                heapq.heappush(nearest, (-dist, node.doc_id))
            elif -dist > nearest[0][0]:
                heapq.heappushpop(nearest, (-dist, node.doc_id))
            
            # Determine which subtree to search first
            axis = depth % self.dimension
            first, second = (node.left, node.right) if query_vector[axis] < node.vector[axis] else (node.right, node.left)
            
            # Search the first subtree
            _search(first, depth + 1)
            
            # Check if we need to search the second subtree
            if len(nearest) < k or abs(query_vector[axis] - node.vector[axis]) < -nearest[0][0]:
                _search(second, depth + 1)
        
        _search(self.root, 0)
        
        # Return the document IDs
        return [doc_id for _, doc_id in sorted(nearest, reverse=True)]
