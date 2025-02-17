import os
import sys
from langchain_community.llms import OpenAI

# Retrieve API key
OPENAI_API_KEY = os.environ.get('OPENAI_APIKEY')


def run_customer_support(query: str) -> str:
    print(f"🔍 Received Query in Agent: {query}")  # Confirm function is called
    sys.stdout.flush()

    try:
        # Check if API key is available
        if not OPENAI_API_KEY:
            raise ValueError(
                "❌ ERROR: OpenAI API Key is missing! Make sure it's set in Replit Secrets."
            )

        # Initialize OpenAI LLM
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        print("✅ OpenAI LLM Successfully Initialized")  # Debug OpenAI
        sys.stdout.flush()

        # Test a basic OpenAI request before using LangChain
        openai_response = llm.invoke("Hello, who are you?")
        print(f"📝 OpenAI Test Response: {openai_response}"
              )  # Debug OpenAI response
        sys.stdout.flush()

        return openai_response

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")  # Show errors in logs
        sys.stdout.flush()
        return "An error occurred while processing your request."
