import os
import sys
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings

# Retrieve API Keys
OPENAI_API_KEY = os.environ.get('OPENAI_APIKEY')
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
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Connect to existing Pinecone index
        vector_db = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        # Use Pinecone for FAQ retrieval
        results = vector_db.similarity_search(query, k=3)
        if results:
            context = "\n".join([doc.page_content for doc in results])
            query = f"Based on the following FAQs, answer concisely:\n\n{context}\n\nUser Query: {query}"
            print("✅ Pinecone retrieved relevant FAQ context.")
        else:
            print("⚠️ No relevant FAQs found in Pinecone.")
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
