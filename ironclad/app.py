from flask import Flask, request, jsonify
import sys
import os
import pandas as pd
from pipeline import Pipeline
from PIL import Image
import io
import time


app = Flask(__name__)
pipeline = Pipeline(pretrained='vggface2', index_type='HNSW', metric='cosine')
evaluation_history = []

query_count = 0  # Counter for throughput calculation
start_time = time.time()  # Start time for throughput calculation
top_k_match_count = 0  # Count of successful top-k matches


# Initialize pipeline gallery and index
# module_path = os.path.abspath(os.path.join('..'))
# sys.path.append(module_path)
gallery_index_path = "storage\\multi_image_gallery" # Gallery path
probe_folder_path = ".simclr_resources\\probe" # Probe path

# Load the precaculated FAISS index and metadata for the model
faiss_index_path_load = os.path.join('storage', 'catalog_vggface2', 'faiss_index.bin')
metadata_path_load = os.path.join('storage', 'catalog_vggface2', 'metadata.pkl')

pipeline.index.load(faiss_index_path_load, metadata_path_load)

@app.route('/')
def home():
    return "Welcome to the visual search API! Use the /identify endpoint to start the authentication process."


@app.route('/identify', methods = ['POST'])
def identify():
    global query_count, top_k_match_count

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Start time for query latency measurement
    query_start_time = time.time()

    try:
        # Process the image
        img = Image.open(io.BytesIO(file.read()))

        # Get 'k' parameter from query string (default to 5 if not provided)
        k = int(request.args.get('k', 5))

        # Expected label for top-k match rate
        ground_truth_label = request.form.get('label', None)

        # Use the pipeline to find top-k identities
        results = pipeline.search_gallery(img, k)

        # Check for correct match in top-k results if ground truth label is provided
        correct_match = any(result['name'] == ground_truth_label for result in results) if ground_truth_label else False
        if correct_match:
            top_k_match_count += 1  # Increment top-k match count for successful match

        # Calculate query latency
        query_latency = time.time() - query_start_time

        # Update throughput (queries per second)
        query_count += 1
        throughput = query_count / (time.time() - start_time)

        # Record the evaluation in history
        evaluation_history.append({
            "query": file.filename,
            "top_k_results": results,
            "query_latency": query_latency,
            "throughput": throughput,
            "top_k_match_rate": top_k_match_count / query_count if query_count > 0 else 0
        })

        return jsonify({
            "query": file.filename,
            "top_k_results": results,
            "query_latency": query_latency,
            "throughput": throughput,
            "top_k_match_rate": top_k_match_count / query_count if query_count > 0 else 0
        }), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add', methods=['POST'])
def add_identity():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the image
        img = Image.open(io.BytesIO(file.read()))
        identity_name = request.form.get("name", file.filename)  # Get identity name from form or use filename

        # Generate embedding for the new identity
        embedding_vector = pipeline.__encode(img)

        # Prepare metadata for the new identity
        metadata = {
            "name": identity_name,
            "filename": file.filename,
            "embedding": embedding_vector
        }

        # Add the embedding and metadata to the FAISS index
        pipeline.index.add_embeddings(embedding_vector, metadata)

        # Record the new addition in evaluation history
        evaluation_history.append({
            "action": "add",
            "identity": identity_name,
            "filename": file.filename
        })

        return jsonify({
            "message": f"Identity '{identity_name}' added to the gallery.",
            "filename": file.filename
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Return historical predictions/searches and their pertinent details."""
    return jsonify({"history": evaluation_history}), 200

    
if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

    # To run, input below in terminal:
    # python app.py
