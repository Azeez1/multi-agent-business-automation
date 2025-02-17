import os
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Retrieve API Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # Example: "us-west1-gcp"
INDEX_NAME = "customer-support"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI's embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"))

# Connect to Pinecone index
pinecone_index = pc.Index(INDEX_NAME)
