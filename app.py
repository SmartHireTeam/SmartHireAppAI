from flask import Flask, request, jsonify
from scipy.spatial.distance import cdist
from openai import OpenAI
from tqdm import tqdm
from scipy.stats import rankdata
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import re, os, logging, numpy as np
from process_data import process_data
from azure.storage.blob import BlobServiceClient
import nltk
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Create 'nltk_data' directory if it doesn't exist
if not os.path.exists('nltk_data'):
    os.makedirs('nltk_data')

# Download 'stopwords' and 'wordnet' corpora to 'nltk_data' directory
nltk.download('stopwords', download_dir='nltk_data')

# Add 'nltk_data' directory to NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))


# Set up logging
logging.basicConfig(filename='app.log',level=logging.INFO)

connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
openai_api_key = os.getenv('OPENAI_API_KEY')

blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
# Load your precomputed embeddings
precomputed_embeddings = np.load('test_data_embeddings.npy')
client = OpenAI(api_key=openai_api_key)

resume_paths = []

def populate_resume_paths():
    # Populate the resume paths
    resume_dir = './test_data/'
    for filename in os.listdir(resume_dir):
        if filename.endswith('.pdf'):
            resume_paths.append(filename)

def download_documents_from_azure_blob(container_name, download_path):
    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    print(container_client.list_blob_names())

    blob_list = container_client.list_blobs()

    for blob in blob_list:
        file_path = os.path.join(download_path, blob.name)
        blob_client = blob_service_client.get_blob_client(container_name, blob.name)
        with open (file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    return resume_paths

# download_documents_from_azure_blob("data", "./azure_data/")

def generate_keyword_embeddings(texts):
    inputs = texts if isinstance(texts, list) else [texts]
    embeddings = []

    for text_chunk in tqdm(inputs, desc='Generating embeddings for keywords'):
        response = client.embeddings.create(
                    input=text_chunk,
                    model="text-embedding-3-small",
                    )
        embeddings.append(response.data[0].embedding)
    return embeddings

def pre_process_text(text):
    # Convert to lower case
    text = str(text).lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin tokens
    return ' '.join(tokens)


@app.route('/search/resumes', methods=['POST'])
def match_document():
    content = request.json
    text = content["context"]
    inputPath = content['inputPath']
    threshold = content.get('threshold', 0.7)
    no_of_matches = content.get('noOfMatches', 3)
    processed_text = pre_process_text(text)  # Ensure you have a preprocess_text function

    # downloaded_documents = download_documents(inputPath)
    # precomputed_embeddings =  process_data("./test_data/files/")

    populate_resume_paths()

    keywords_embeddings = generate_keyword_embeddings(processed_text)
    # keywords_embeddings = model.encode(processed_text, show_progress_bar=True)

    keywords_embeddings_norm = normalize(keywords_embeddings)
    precomputed_embeddings_norm = normalize(precomputed_embeddings)

    similarities = 1 - cdist(keywords_embeddings_norm, precomputed_embeddings_norm, metric='cosine')[0]
    print(similarities)
    percentile_ranks = rankdata(similarities, method='max') / len(similarities) * 100
    print(percentile_ranks)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def adjust_scale(scores):
        return (scores - np.mean(scores)) / (np.std(scores))
    
    scaled_similarities = sigmoid(adjust_scale(similarities))
    print(scaled_similarities.shape)

    valid_indices = np.where(scaled_similarities >= threshold)[0]
    valid_similarities = scaled_similarities[valid_indices]

    print(valid_indices, valid_similarities)

    top_indices = valid_indices[np.argsort(valid_similarities)[::-1][:no_of_matches]]
    top_scores = valid_similarities[np.argsort(valid_similarities)[::-1][:no_of_matches]]

    print(top_indices, top_scores)
    print(resume_paths)
    results = []

    for index, score in zip(top_indices, top_scores):
        results.append({
            "id": int(index),
            "score": float(score),
            "document": resume_paths[index]
        })

    return jsonify({
        "status": "success",
        "count": len(results),
        "metadata": {"confidenceScore": float(np.mean(top_scores)) if results else 0},
        "results": results
    })

@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({"error": error}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
