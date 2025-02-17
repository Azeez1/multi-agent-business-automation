import os
import sys
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone_setup import pinecone_index  # Import Pinecone setup

# Retrieve API Keys
OPENAI_API_KEY = os.environ.get('OPENAI_APIKEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')
INDEX_NAME = "customer-support"

# ✅ API Key Debugging
print("🔑 Checking API Keys...")
if OPENAI_API_KEY:
    print(
        "✅ OpenAI API Key: ✔️ Loaded Successfully (Not Displayed for Security)"
    )
else:
    print(
        "❌ ERROR: OpenAI API Key is MISSING! Set it in environment variables.")

if PINECONE_API_KEY:
    print("✅ Pinecone API Key: ✔️ Loaded Successfully")
else:
    print(
        "❌ ERROR: Pinecone API Key is MISSING! Set it in environment variables."
    )

if PINECONE_ENV:
    print(f"✅ Pinecone Environment: {PINECONE_ENV}")
else:
    print(
        "❌ ERROR: Pinecone Environment is MISSING! Set it in environment variables."
    )

sys.stdout.flush()


def initialize_vector_db():
    """Load or create a vector database for FAQ retrieval using Pinecone."""
    try:
        print("🔍 Initializing vector database...")
        sys.stdout.flush()

        # Check if FAQ file exists
        if not os.path.exists("faq_data.txt"):
            print(
                "❌ ERROR: faq_data.txt file is missing! Please create one with sample FAQs."
            )
            sys.stdout.flush()
            return None

        loader = TextLoader("faq_data.txt")
        documents = loader.load()

        if not documents:
            print(
                "❌ ERROR: faq_data.txt is empty! Please add FAQs for retrieval."
            )
            sys.stdout.flush()
            return None

        print("✅ FAQ data loaded successfully.")
        sys.stdout.flush()

        # Split text into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Convert text into embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Store in Pinecone
        vectorstore = Pinecone.from_documents(docs,
                                              embeddings,
                                              index_name=INDEX_NAME)
        print("✅ Vector database initialized successfully.")
        sys.stdout.flush()

        return vectorstore
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Pinecone vector DB - {e}")
        sys.stdout.flush()
        return None


# Load Pinecone Vector DB
vector_db = initialize_vector_db()

# ✅ Pinecone Check
if not vector_db:
    print("❌ ERROR: Pinecone vector database is not initialized!")
    sys.stdout.flush()


def run_customer_support(query: str) -> str:
    """Handles customer queries using OpenAI & Vector Search via Pinecone"""
    print(f"🔍 Received Query: {query}")
    sys.stdout.flush()

    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API Key is missing. Please configure environment variables."

    try:
        # Initialize OpenAI LLM
        print("🔄 Initializing OpenAI LLM...")
        sys.stdout.flush()
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        print("✅ OpenAI LLM initialized successfully.")
        sys.stdout.flush()

        # Use Pinecone for FAQ retrieval if available
        if vector_db:
            results = vector_db.similarity_search(query, k=3)
            if results:
                context = "\n".join([doc.page_content for doc in results])
                query = f"Based on the following FAQs, answer concisely:\n\n{context}\n\nUser Query: {query}"
                print("✅ Pinecone retrieved relevant FAQ context.")
            else:
                print("⚠️ No relevant FAQs found in Pinecone.")
                query = f"User Query: {query}"
        else:
            print("❌ ERROR: Pinecone vector database is not available.")
            query = f"User Query: {query}"

        # Generate response
        response = llm.invoke(query)
        print(f"📝 AI Response: {response}")
        sys.stdout.flush()

        return response

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.stdout.flush()
        return "⚠️ An error occurred while processing your request. Please try again."


if __name__ == "__main__":
    test_query = "Where is my order?"
    response = run_customer_support(test_query)
    print(f"🤖 Bot Response: {response}")
