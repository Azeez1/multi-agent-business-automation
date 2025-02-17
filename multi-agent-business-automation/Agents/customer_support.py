import os
import sys
import pinecone
from langchain_community.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Retrieve API Keys
OPENAI_API_KEY = os.environ.get('OPENAI_APIKEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')  # Example: "us-west1-gcp"
INDEX_NAME = "customer-support"  # Pinecone index name



# Correctly initialize Pinecone

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)econe.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Ensure the index exists
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME,
                          dimension=1536)  # OpenAI's embedding size

# Connect to Pinecone index
pinecone_index = pinecone.Index(INDEX_NAME)


# Load Vector Database with FAQs
def initialize_vector_db():
    """Load or create a vector database for FAQ retrieval using Pinecone."""
    try:
        # Load FAQ documents
        loader = TextLoader("faq_data.txt")  # Ensure this file exists
        documents = loader.load()

        # Split text into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Convert text into embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Store in Pinecone
        vectorstore = Pinecone.from_documents(docs,
                                              embeddings,
                                              index_name=INDEX_NAME)
        return vectorstore
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Pinecone vector DB - {e}")
        sys.stdout.flush()
        return None


# Load Pinecone Vector DB
vector_db = initialize_vector_db()


def run_customer_support(query: str) -> str:
    """Handles customer queries using OpenAI & Vector Search via Pinecone"""
    print(f"🔍 Received Query: {query}")
    sys.stdout.flush()

    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API Key is missing. Please configure environment variables."

    try:
        # Initialize OpenAI LLM
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_APIKEY)

        # Use Pinecone for FAQ retrieval if available
        if vector_db:
            results = vector_db.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in results])
            query = f"Based on the following FAQs, answer concisely:\n\n{context}\n\nUser Query: {query}"

        # Generate response
        response = llm.invoke(query)
        print(f"📝 AI Response: {response}")
        sys.stdout.flush()

        return response

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.stdout.flush()
        return "⚠️ An error occurred while processing your request. Please try again."
