import os
import json
import pinecone
import openai
import time

# Specify the index name
index_name = "buddhism-sutta"

# Initialize Pinecone
pinecone.init(api_key="58e4f46c-e4b8-4dfd-9ea9-42867c1e59c5", environment="us-central1-gcp")

# Create a new vector index only if it doesn't already exist
if index_name not in pinecone.list_indexes():
    print("Creating index...")
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
    )
    print("Index created.")

index = pinecone.Index(index_name)

# Generate Embeddings and Upload
def generate_and_upload(file_name, chunks):
    sutta_id = file_name.split('_')[0]  # Extract the sutta ID from filename
    for i, text in enumerate(chunks):
        print(f"Generating embedding for {sutta_id}-{i+1}...")
        embeddings = get_embedding(text)
        print(f"Uploading {sutta_id}-{i+1}...")
        index.upsert([(f"{sutta_id}-{i+1}", embeddings, {"sutta": sutta_id, "text": text})])
        print(f"Uploaded {sutta_id}-{i+1}.")

def get_embedding(text, model="text-embedding-ada-002"):
   embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   print("Waiting 20 seconds...")
   time.sleep(20)
   return embedding

def split_text(text, token_limit=20000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) <= token_limit:
            current_chunk.append(word)
            current_length += len(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    chunks.append(' '.join(current_chunk))  # Add the last chunk
    return chunks

# Load text from files in the same directory
for file_name in os.listdir():
    if file_name.endswith('.json'):
        print(f"Processing {file_name}...")
        with open(file_name, 'r') as file:
            text = file.read()  # Get the entire file content as text
            chunks = split_text(text)
            print(f"Uploading chunks for {file_name}...")
            generate_and_upload(file_name, chunks)
        print(f"Deleting {file_name}...")
        os.remove(file_name)
        print(f"Deleted {file_name}.")

