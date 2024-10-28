"""This file contains method to extract the embedding vector from a preprocessed image"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from extraction.preprocess import Preprocessing
from extraction.embedding import Embedding
from retrieval.indexing import FaissIndex
from retrieval.search import FaissSearch
import faiss
import PIL
from PIL import Image
import numpy as np
import pickle
import torch

class Pipeline:
    def __init__(self, pretained, index_type='brute_force', metric='euclidean', **kwargs):
        """
        Initialize the pipeline, setting up FAISS index, model, and other components.
        :param kwargs: All keyword arguments passed to configure the pipeline.
        """
        # Check if CUDA is available and select the appropriate device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set default values and allow kwargs to override them
        self.pretrained = pretained
        self.index_type = index_type
        self.metric = metric

        self.device = kwargs.get('device', 'cpu')
        # self.vector_dimension = kwargs.get('vector_dimension', 128)
        self.image_size = kwargs.get('image_size', 160)
        self.p = kwargs.get('p', 3)

        # Initialize embedding and model
        self.embed = Embedding(self.pretrained, self.device)
        self.model = self.embed.model

        # Initialize FAISS index, preprocess, and search components
        self.vector_dimension = None
        self.index = None
        self.preprocessing = Preprocessing(self.image_size)
        self.searching = None # Initialize later in precompute_and_save() method

        self.index_params = kwargs  # Store remaining kwargs for later customization if needed
        self.metadata = []  # Store metadata for each point


    def __encode(self, image):
        """ Get the embedding vector from the processed image"""
        embedding_vector = self.embed.encode(image)
        embedding_vector = np.array(embedding_vector).reshape(1, -1)
        return embedding_vector


    def __precompute(self, gallery_directory: str):
        """Extract embeddings from all images in the gallery and store in a faiss database"""

        # Read each image for each person in the gallery
        for person_name in os.listdir(gallery_directory):
            person_folder = os.path.join(gallery_directory, person_name)

            for file_name in os.listdir(person_folder):
                if file_name.endswith(('.jpg', '.png', '.jpeg')) and not file_name.startswith('._'):
                    image_path = os.path.join(person_folder, file_name)

                    try:
                        with Image.open(image_path) as img:

                            # 1. Preprocess each image
                            processed_image = self.preprocessing.process(img)

                            # 2. Get embedding vector from the preprocessed image
                            embedding_vector = self.__encode(processed_image)

                            # 3. Dynamically set vector_dimension when extract the first embedding
                            if self.vector_dimension is None:
                                self.vector_dimension = embedding_vector.shape[1]
                                # Initialize the FAISS index now
                                self.index = FaissIndex(self.vector_dimension, self.index_type)
                                self.index._create_index()

                            # 3. Prepare metadata for this embedding
                            metadata = {
                                'name': [person_name],
                                'filename': [file_name],
                                'embedding': [embedding_vector],
                            }

                            # 4. Add the embedding vector to the FAISS index
                            self.index.add_embeddings(embedding_vector, metadata)

                    except PIL.UnidentifiedImageError:

                        print(f"Skipping file {image_path}: UnidentifiedImageError")

        return self.index


    def __save_embeddings(self, faiss_path: str, metadata_path: str):
        """Store the embeddings in a FAISS's serialized binary format"""
        if not os.path.exists(faiss_path):
            os.makedirs(faiss_path)
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)

        faiss_index_path = os.path.join(faiss_path, 'faiss_index.bin')
        metadata_file_path = os.path.join(faiss_path, 'metadata.pkl')

        # Save faiss index and metadata
        self.index.save(faiss_index_path, metadata_file_path)


    def precompute_and_save(self, gallery_directory: str, faiss_path: str, metadata_path: str):
        """Public method to call private __precompute and __save_embeddings internally"""
        self.__precompute(gallery_directory)
        self.__save_embeddings(faiss_path, metadata_path)
        # Initialize FaissSearch after the index has been fully created and populated
        self.searching = FaissSearch(self.index, self.metric, self.p)
        print(f"Precomputed and saved embeddings for the gallery images.")

    
    def precompute(self, gallery_directory: str):
        """Public method to call private __precompute and __save_embeddings internally"""
        self.__precompute(gallery_directory)
        # Initialize FaissSearch after the index has been fully created and populated
        self.searching = FaissSearch(self.index, self.metric, self.p)
        print(f"Precomputed the gallery images.")


    def search_gallery(self, probe, k):
        """Search the k-nearest-neighbors of a probe
        return: a list of k individuals' names, the source image filename, and the vector embedding;
        """
        # Preprocess the probe
        preprocessed_probe = self.preprocessing.process(probe)
        # Get the embedding of the preprocessed probe
        embedding_vector = self.__encode(preprocessed_probe)
        distances, indices, metadata_results = self.searching.search(embedding_vector, k=k)

        # Prepare the result list
        results = []
        for i in range(k):
            distance = distances[i] if self.metric == 'minkowski' else distances[0][i]
            result = {
                'index': indices[0][i],  # Extract the index of the neighbor
                'distance': distance,
                'name': metadata_results[i]['name'],
                'filename': metadata_results[i]['filename'],
                'embedding': metadata_results[i]['embedding']
            }
            results.append(result)
        
        return results


    def show_all_indexed_entries_with_distance(self, probe):
        """
        Show all indexed entries along with their distances from the probe embedding in the format:
        Index: 5, Distance: 0.2593, Name: ['Kalpana_Chawla'], Filename: ['Kalpana_Chawla_0001.jpg']
        
        This function uses the search method from FaissSearch to compute distances to all indexed entries.
        """
        # Preprocess the probe
        preprocessed_probe = self.preprocessing.process(probe)
        # Get the embedding of the preprocessed probe
        probe_embedding = self.__encode(preprocessed_probe)

        # Set k to the total number of entries in the FAISS index
        k = self.index.index.ntotal  # Get the total number of indexed vectors

        # Perform a search for all indexed entries
        distances, indices, metadata_results = self.searching.search(probe_embedding, k=k)

        # Print the formatted output for each result
        for i in range(k):
            print(f"Index: {indices[0][i]}, Distance: {distances[0][i]:.4f}, Name: {metadata_results[i]['name']}, Filename: {metadata_results[i]['filename']}")


if __name__ == "__main__":
    # SAMPLE TEST
    # Initialize pipeline
    pipeline = Pipeline('casia-webface', index_type='IVF', metric='minkowski')

    # Precomput and save the gallery index
    gallery_path = "storage\\sample_gallery"

    pipeline.precompute("storage/sample_gallery")

    # Read in the sample probe
    probe_folder_path = "simclr_resources\\probe"
    sample_probe_name = "Aaron_Sorkin\\Aaron_Sorkin_0002.jpg"
    sample_probe_path = os.path.join(probe_folder_path, sample_probe_name)
    sample_probe = Image.open(sample_probe_path)

    print(f"For probe {sample_probe_name}, The nearest neighbors are:")
    results = pipeline.search_gallery(sample_probe, k=3)
    for result in results:
        print(f"Index: {result['index']}, Distance: {result['distance']:.4f}, Name: {result['name']}, Filename: {result['filename']}")

    # Read in gallery index and metadata
    # faiss_index_path = os.path.join('storage', 'sample_catalog', 'faiss_index.bin')
    # metadata_path = os.path.join('storage', 'sample_catalog', 'metadata.pkl')
    # pipeline.index.load(faiss_index_path, metadata_path)

    # for result in results:
    #     print(f"Index: {result['index']}, Distance: {result['distance']:.4f}, Name: {result['name']}, Filename: {result['filename']}")

    # print(f"\nAll indexed entries are:")
    # pipeline.show_all_indexed_entries_with_distance(sample_probe)

