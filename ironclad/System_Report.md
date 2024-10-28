# IronClad Object Det

## Table of Contents
1. [System Design](#system-design)
2. [Metrics Definition](#metrics-definition)
3. [Analysis of System Parameters and Configurations](#analysis-of-system-parameters-and-configurations)

---
## System Design

### 1. Extraction Service
The Extraction Service is responsible for preprocessing the input images to ensure they comply with the embedding model’s specifications. It extracts embedding vectors, which are standardized vectors that serve as compact representations for nearest neighbor search, from these preprocessed images.

### 2. Retrieval Service
The Retrieval Service indexes the catalog storage images to enable nearest neighbor searches. The system allows retrievals across images, returning the top-k nearest neighbors based on similarity. The k value is user-defined, which helps tailor search granularity per use case.

### 3. Interface Service
The Interface Service provides a streamlined way to interact with the system’s core functionalities through a unified API. It coordinates image embedding, indexing, and retrieval by managing the Extraction Service and Retrieval Service processes within the Pipeline class. This class includes methods for embedding individual probe images (__encode), precomputing embeddings for the gallery (__precompute), saving these embeddings in a FAISS index (__save_embeddings), and performing k-nearest-neighbor searches (search_gallery). Additionally, the service includes Flask endpoints to trigger the identification process (/identify), add new identities incrementally (/add), and retrieve historical searches (/history). By encapsulating these operations, the Interface Service ensures a seamless and efficient flow, handling image preprocessing, embedding extraction, index storage, and retrieval as needed for practical system deployment.

---

## Metrics Definition

### Offline Metrics
Offline metrics are used to evaluate system performance in controlled, offline environments. They include:

- **Mean Reciprocal Rank (MRR):** Measures how well the system ranks the true match for a probe in its retrieval list, providing insight into rank accuracy for known matches.
- **Precision@k:** Measures the accuracy of top-k results, providing insight into search precision for each query.
- **Mean Average Precision (mAP):** Assesses retrieval quality by averaging precision at each relevant result across queries, especially useful for ranking tasks.

These metrics are evaluated offline to iteratively improve the system and measure its search precision and ranking quality under various settings.

### Online Metrics
Online metrics monitor the system’s real-time performance in production. These include:

- **Query Latency:** Measures the time taken to return results from a query and time taken on precomputation of the gallery. Lower latency indicates faster response times for users.

---

## Analysis of System Parameters and Configurations

This section evaluates significant design decisions made across the Extraction and Retrieval Services.

### 1. Embedding Model Choice (Extraction Service)
**Decision:** We evaluated both `vggface2` and `casia-webface` models to determine which provides more accurate representations for our data. The chosen model directly influences the quality of embeddings and affects retrieval precision.

**Evaluation Method:** MRR, Precision@k, and mAP were used to benchmark each model's accuracy and rank performance. Results indicated that `vggface2` achieved higher MRR and mAP scores under the same index, metric and k values, suggesting it provided embeddings more aligned with our use case. This model is also used as the default model in the flask app.

![image info](.\notebooks\pictures\model_comparison.png)

**Impact:** Using `vggface2` enables more accurate similarity searches, critical for the system's retrieval precision.

### 2. Image Preprocessing Techniques (Extraction Service)
**Decision:** Experimented with normalization, resizing, and transformations such as horizontal flipping and brightness adjustment.

**Evaluation Method:** MRR and query latency were assessed across original and transformed images to evaluate retrieval performance. Each transformation’s effect on the rank of the correct identity in the top-k results was tracked to ensure model robustness under different conditions.

![image info](.\notebooks\pictures\image_transforms.png)
![image info](.\notebooks\pictures\image_transforms_2.png)

For this individual probe experiment, Baseline (Original) and Rotate Small maintain a high rank and multiple correct returns. These transformations have minimal impact on the model’s ability to correctly identify the probe. Brightness Increase results in the absence of the correct name from the top-k results, demonstrating a severe degradation in retrieval accuracy.

Overall MRR on looping all probes under different transformations are also experimented and evaluated:

![image info](.\notebooks\pictures\image_transforms_3.png)

From the data, Random Crop (Small), Random Crop (Large), Gaussian Blur (Strong), Rotate Large, and Brightness Increase are the most impactful transformations leading to the lowest MRR values.

**Impact:** Proper preprocessing leads to more consistent embedding representations, which directly improves retrieval accuracy.

### 3. Choice of Index Type (Retrieval Service)
**Decision:** Different index types are evaluated to balance memory usage, retrieval speed, and accuracy. The selected index impacts how well the system scales to billions of images.

**Evaluation Method:** `Brute Force`, `IVF`, `HNSW`, and `IVFSQ` indexes are evaluated for Model: vggface2, Metric: Cosine and k: 5. Query latency, MRR, precision@k, and mAP are used to compare the trade-offs. Note that latency is calculated around the operations to calculate mrr, precision@k, and mAP, excluding time on precomputation of the gallery images due to precomputation is a one-time thing. `HNSW` was chosen for large-scale indexing due to its balance of speed and accuracy, even though while `Brute Force` indexing scored slightly higher in MRR and mAP in this small dataset (currently 1000 profiles in gallery with around 1 to 5 images per profile).

![image info](.\notebooks\pictures\index_1.png)
![image info](.\notebooks\pictures\index_2.png)
![image info](.\notebooks\pictures\index_3.png)

**Impact:** Choosing `HNSW` ensures faster search times with acceptable accuracy, meeting the system’s scalability requirements.

### 4. Similarity Metric Selection (Retrieval Service)
**Decision:** Tested cosine, euclidean, and minkowski distance metrics for nearest neighbor search.

**Evaluation Method:** MRR, Precision@k, and mAP were evaluated to identify which metric best aligns with the embedding model. After checking, the distance metrics make very little effect on this dataset with all other specifications remain the same (model, index, and k value).

![image info](.\notebooks\pictures\distance_metric_comparison.png)

**Impact:** Choosing cosine as the default distance metric for the model implemented in flask app.

### 5. Setting k in Top-k Search (Retrieval Service)
**Decision:** Various k values (k=1-10) were tested to determine the optimal search depth for retrieving relevant images.

**Evaluation Method:** The k value was evaluated for its influence on MRR and mAP for Model: vggface2, Index Type: HNSW, Metric: Cosine, balancing retrieval accuracy with processing load. The below graph indicates that MRR started to increase very slowly when k passes 5, and mAP scores the highest when k=5. Therefore, in the flask app, the model asks user for a k number and default to 5 if not provided.

![image info](.\notebooks\pictures\mrr_diffk.png)
![image info](.\notebooks\pictures\map_diffk.png)

**Impact:** An optimal k value k=5 ensures precise retrieval without overloading the system, resulting in efficient and accurate searches.

---

### Summary
The system’s performance has been enhanced through careful consideration of embedding models, index types, similarity metrics, and preprocessing techniques. Future iterations may focus on real-time adjustments and refinements based on observed online metrics.
