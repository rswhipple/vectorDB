import numpy as np
from scipy.spatial import KDTree

class VectorDB:
    def __init__(self, dim, num_hash_tables=5, hash_size=32):
        """
        Initializes the vector database.
        
        Parameters:
        - dim: dimensionality of the input vectors.
        - num_hash_tables: number of LSH tables (each with its own set of random projection hyperplanes).
        - hash_size: number of random hyperplanes per LSH table.
        """
        self.dim = dim
        self.vectors = []   # list to store the normalized vectors
        self.ids = []       # corresponding vector IDs
        self.kd_tree = None # KDTree, built on demand
        
        # LSH parameters
        self.num_hash_tables = num_hash_tables
        self.hash_size = hash_size
        # Each LSH table is a dictionary mapping a hash (tuple of bits) to a list of indices
        self.lsh_tables = [ {} for _ in range(num_hash_tables) ]
        # For each table, generate random hyperplanes (a matrix of shape [hash_size, dim])
        self.hash_planes = [ np.random.randn(hash_size, dim) for _ in range(num_hash_tables) ]

    def insert(self, vector, id):
        """
        Inserts a new vector with an associated id.
        The vector is first normalized.
        Updates both the LSH tables and invalidates the KDTree.
        """
        # Normalize vector
        norm_vector = vector / np.linalg.norm(vector)
        self.vectors.append(norm_vector)
        self.ids.append(id)
        
        # Update each LSH table with the new vector index
        for table, planes in zip(self.lsh_tables, self.hash_planes):
            hash_code = self._compute_hash(norm_vector, planes)
            if hash_code not in table:
                table[hash_code] = []
            table[hash_code].append(len(self.vectors) - 1)
        
        # Invalidate the KDTree since the underlying data has changed
        self.kd_tree = None

    def _compute_hash(self, vector, planes):
        """
        Computes the hash code of a vector using a set of random hyperplanes.
        Returns the hash as a tuple of 0s and 1s.
        """
        projections = np.dot(planes, vector)
        hash_bits = projections > 0  # boolean array: True if projection positive, False otherwise
        return tuple(hash_bits.astype(int))
    
    def build_kd_tree(self):
        """
        Builds (or rebuilds) the KDTree for the current set of vectors.
        """
        if self.vectors:
            data = np.array(self.vectors)
            self.kd_tree = KDTree(data)

    def query(self, query_vector, num_neighbors=5, method='lsh'):
        """
        Queries the database for nearest neighbors to the query_vector.
        
        Parameters:
        - query_vector: the vector to search with (will be normalized).
        - num_neighbors: number of nearest neighbors to return.
        - method: 'lsh' for approximate search using LSH,
                  'kd' for exact search using the KDTree.
        
        Returns:
        A tuple (result_ids, distances) where result_ids is a list of vector IDs
        and distances is an array of distances from the query.
        """
        norm_query = query_vector / np.linalg.norm(query_vector)
        
        if method == 'kd':
            # Build the KDTree if it hasn't been built yet.
            if self.kd_tree is None:
                self.build_kd_tree()
            dists, indices = self.kd_tree.query(norm_query, k=num_neighbors)
            # Ensure indices is a list (if k=1, KDTree returns a scalar)
            if np.isscalar(indices):
                indices = [indices]
            result_ids = [self.ids[i] for i in indices]
            return result_ids, dists
        
        elif method == 'lsh':
            # Gather candidate indices from all LSH tables.
            candidate_indices = set()
            for table, planes in zip(self.lsh_tables, self.hash_planes):
                hash_code = self._compute_hash(norm_query, planes)
                if hash_code in table:
                    candidate_indices.update(table[hash_code])
            
            # If no candidates are found, fall back to KDTree search.
            if not candidate_indices:
                return self.query(norm_query, num_neighbors, method='kd')
            
            # Compute Euclidean distances for the candidate vectors.
            candidates = list(candidate_indices)
            candidate_vectors = np.array([self.vectors[i] for i in candidates])
            distances = np.linalg.norm(candidate_vectors - norm_query, axis=1)
            sorted_indices = np.argsort(distances)[:num_neighbors]
            result_ids = [self.ids[candidates[i]] for i in sorted_indices]
            return result_ids, distances[sorted_indices]
        
        else:
            raise ValueError("Method must be either 'lsh' or 'kd'.")

# Example usage:
if __name__ == "__main__":
    # Create a vector database for 128-dimensional vectors.
    db = VectorDB(dim=128, num_hash_tables=5, hash_size=32)
    
    # Insert 100 random vectors.
    for i in range(100):
        vec = np.random.randn(128)
        db.insert(vec, id=f"vec_{i}")
    
    # Create a random query vector.
    query_vec = np.random.randn(128)
    
    # Query using LSH (approximate search)
    ids_lsh, dists_lsh = db.query(query_vec, num_neighbors=5, method='lsh')
    print("LSH Query Results:")
    for id_val, dist in zip(ids_lsh, dists_lsh):
        print(f"ID: {id_val}, Distance: {dist:.4f}")
    
    # Query using KDTree (exact search)
    ids_kd, dists_kd = db.query(query_vec, num_neighbors=5, method='kd')
    print("\nKDTree Query Results:")
    for id_val, dist in zip(ids_kd, dists_kd):
        print(f"ID: {id_val}, Distance: {dist:.4f}")
