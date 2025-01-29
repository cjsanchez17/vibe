from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import numpy as np
from gensim.models import KeyedVectors
import os
import pickle

# AWS S3 configuration
S3_FILES = {
    "./data/tag_vector_quantized.pt": "tag_vector_quantized.pt",
    "./data/tag_list.npy": "tag_list.npy",
    "./data/converted_music_embeddings.bin": "converted_music_embeddings.bin",
    "./data/wiki-news-300d-1M.vec": "wiki-news-300d-1M.vec",
    "./data/track_ids.npy": "track_ids.npy",
    "./data/track_list.npy": "track_list.npy",
    "./data/track_vector_quantized.pt": "track_vector_quantized.pt",

}

# Initialize S3 client

def download_file_from_s3(s3_path, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {s3_path} to {local_path} from S3...")
        s3.download_file(S3_BUCKET_NAME, s3_path, local_path)
        print(f"Downloaded {s3_path} successfully.")

# Download required files from S3
for local_path, s3_filename in S3_FILES.items():
    download_file_from_s3(s3_filename, local_path)

app = FastAPI()

# CORS Configuration to allow frontend and Express backend to interact
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://gsawyergarrett-cjsanchez17.onrender.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use ./data/ directory for storing large files
TAG_VECTOR_FILE = "./data/tag_vector_quantized.pt"
TAG_LIST_FILE = "./data/tag_list.npy"
GENSIM_MODEL_PATH = "./data/wiki-news-300d-1M.vec"
REDDIT_MODEL_FILE = "./data/converted_music_embeddings.bin"

def load_preprocessed_data():
    global track_vector, track_list, id2url, track_ids
    track_vector = torch.load("./data/track_vector_quantized.pt", map_location=torch.device('cpu'))
    track_vector = torch.nn.functional.normalize(track_vector)
    
    # Load track list
    track_list = np.load("./data/track_list.npy", allow_pickle=True).tolist()
    
    # Load track IDs
    track_ids = np.load("./data/track_ids.npy", allow_pickle=True).tolist()

load_preprocessed_data()


# Load the models and embeddings from persistent disk storage
tag_vector = torch.load(TAG_VECTOR_FILE, map_location=torch.device('cpu'))  # Load .pt file safely
tag_list = np.load(TAG_LIST_FILE, allow_pickle=True).tolist()
gensim_vectors = KeyedVectors.load_word2vec_format(GENSIM_MODEL_PATH, binary=False)
model = KeyedVectors.load_word2vec_format(REDDIT_MODEL_FILE, binary=True)

def get_combined_embedding(word):
    """
    Retrieves the embedding of a given word from the tag list or gensim vectors.
    """
    if word in tag_list:
        idx = tag_list.index(word)
        return tag_vector[idx]  # Return from tag vector tensor
    elif word in model:
        return torch.tensor(model[word])
    elif word in gensim_vectors:
        return torch.tensor(gensim_vectors[word])  # Convert gensim embedding to torch tensor
    return None

def process_tokens(query, vector_type):
    """
    return sorted vector indices and scores for either music_retrieval or recommend tags.
    depends on vector_type (i.e. tag_vector or track_vector)
    
    """
    token_list = [i.strip() for i in query.split()]

    # Get embeddings for tokens
    query_embeddings = [get_combined_embedding(token) for token in token_list]
    query_embeddings = [emb for emb in query_embeddings if emb is not None]  # Filter out missing embeddings

    if not query_embeddings:
        return {"query": query, "results": []}

    query_vector = torch.stack(query_embeddings)
    query_vector = torch.nn.functional.normalize(query_vector)  # Normalize
    if query_vector.size(0) > 1:
        query_vector = query_vector.mean(0, True)  # Average if multiple tokens
    
    query_vector = query_vector.to(dtype=vector_type.dtype)

    # Calculate similarity scores
    score_matrix = query_vector @ vector_type.T
    sorted_indices = torch.flip(torch.argsort(score_matrix, dim=1), dims=[1])
    score_matrix = score_matrix.squeeze(0)
    sorted_indices = sorted_indices.squeeze(0)

    # Get top-k results
    return (score_matrix, sorted_indices)
def recommend_tags(query, topk=10):

    # Get top-k results
    score_matrix, sorted_indices = process_tokens(query, tag_vector)
    topk_indices = sorted_indices[:topk]
    results = []
    for i in topk_indices:
        results.append({
            "entity": tag_list[i],
            "score": float(score_matrix[i]),
        })
    return {"query": query, "results": results}

def music_retrieval(query, topk=10):

    score_matrix, sorted_indices = process_tokens(query, track_vector)
    topk_indices = sorted_indices[:topk]
    results = []
    for i in topk_indices:
        track_id = track_list[i]
        results.append({
            "entity": track_list[i],
            "score": float(score_matrix[i]),
        })
    return {"query": query, "results": results}


@app.get("/recommend/")
async def recommend(query: str = Query(..., min_length=1)):
    """
    Handles the recommendation query, returning results from multiple sources.
    """
    faiss_results = music_retrieval(query, topk=20)
    # return [faiss_results, recommend_tags(query)]
    return faiss_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)