import numpy as np

class FaissSearch:
    def __init__(self, faiss_index, metric='euclidean', p=3):
        """
        Initialize the search class with a FaissIndex instance and distance metric.
        
        :param faiss_index: A FaissIndex instance.
        :param metric: The distance metric ('euclidean', 'dot_product', 'cosine', 'minkowski').
        :param p: The parameter for Minkowski distance (p=2 for Euclidean, p=1 for Manhattan).
        """
        self.index = faiss_index.index
        self.metric = metric
        self.p = p  # Minkowski distance parameter
        self.faiss_index = faiss_index

    def search(self, query_vector, k=5):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        
        :param query_vector: The vector to query (numpy array).
        :param k: Number of nearest neighbors to return.
        :return: Distances, indices, and metadata of the nearest neighbors.
        """
        if self.metric == 'cosine':
            # Normalize vectors for cosine similarity
            query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
            distances, indices = self.index.search(query_vector, k)
            # Convert dot product to cosine distance: distance = 1 - cosine similarity
            distances = 1 - distances

        elif self.metric == 'minkowski':
            # Use FAISS to perform an approximate Euclidean search
            distances, indices = self.index.search(query_vector, k)
            # Manually compute Minkowski distance based on the returned vectors
            nearest_vectors = [self.index.reconstruct(int(i)) for i in indices[0]]
            distances = self._compute_minkowski(query_vector[0], nearest_vectors, p=self.p)

        else:
            # Default FAISS search (Euclidean or Dot Product)
            distances, indices = self.index.search(query_vector, k)

        # Retrieve metadata for the nearest neighbors
        metadata_results = [self.faiss_index.get_metadata(int(i)) for i in indices[0]]
        return distances, indices, metadata_results

    def _compute_minkowski(self, query_vector, nearest_vectors, p):
        """
        Compute Minkowski distance between the query vector and the nearest neighbors.
        
        :param query_vector: The query vector (numpy array).
        :param nearest_vectors: List of nearest neighbor vectors (numpy arrays).
        :param p: Parameter for Minkowski distance (p=2 is Euclidean, p=1 is Manhattan).
        :return: List of Minkowski distances.
        """
        distances = []
        for vec in nearest_vectors:
            distance = np.sum(np.abs(query_vector - vec) ** p) ** (1 / p)
            distances.append(distance)
        return distances


if __name__ == "__main__":
    import numpy as np

    from indexing import FaissIndex
    from search import FaissSearch


    # Create some random vectors (10k vectors of dimension 128)
    vectors = np.random.random((10000, 256)).astype('float32')
    metadata = [f"Vector-{i}" for i in range(10000)]
    query_vector = np.random.random((1, 256)).astype('float32')

    # Example 1: IVF with custom nlist (number of clusters)
    print("\nExample 1: IVF Index with Custom nlist")
    faiss_index_ivf = FaissIndex(index_type='IVF', nlist=200)
    faiss_index_ivf.add_embeddings(vectors, metadata=metadata)
    faiss_search_ivf = FaissSearch(faiss_index_ivf, metric='euclidean')
    distances_ivf, indices_ivf, metadata_ivf = faiss_search_ivf.search(query_vector, k=5)
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Metadata: {metadata_ivf[i]}")


    # Example 2: PQ with custom m and bits_per_subquantizer
    print("\nExample 2: PQ Index with Custom m and bits_per_subquantizer")
    faiss_index_pq = FaissIndex(index_type='PQ', m=16, bits_per_subquantizer=6)
    faiss_index_pq.add_embeddings(vectors, metadata=metadata)
    faiss_search_pq = FaissSearch(faiss_index_pq, metric='dot_product')
    distances_pq, indices_pq, metadata_pq = faiss_search_pq.search(query_vector, k=5)
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_pq[0][i]}, Distance {distances_pq[0][i]}, Metadata: {metadata_pq[i]}")


    # Example 3: HNSW with custom hnsw_m (number of neighbors)
    print("\nExample 3: HNSW Index with Custom hnsw_m")
    faiss_index_hnsw = FaissIndex(index_type='HNSW', hnsw_m=48)
    faiss_index_hnsw.add_embeddings(vectors, metadata=metadata)
    faiss_search_hnsw = FaissSearch(faiss_index_hnsw, metric='euclidean')
    distances_hnsw, indices_hnsw, metadata_hnsw = faiss_search_hnsw.search(query_vector, k=5)
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_hnsw[0][i]}, Distance {distances_hnsw[0][i]}, Metadata: {metadata_hnsw[i]}")


    # Example 4: Binary IVF with custom nlist for binary data
    print("\nExample 4: Binary IVF Index with Custom nlist")
    faiss_index_flat = FaissIndex(index_type='brute_force', nlist=50)
    faiss_index_flat.add_embeddings(vectors, metadata=metadata)
    faiss_search_flat = FaissSearch(faiss_index_flat, metric='dot_product')
    distances_flat, indices_flat, metadata_flat = faiss_search_flat.search(query_vector, k=5)
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_flat[0][i]}, Distance {distances_flat[0][i]}, Metadata: {metadata_flat[i]}")