import os
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

# Retrieve API key
OPENAI_API_KEY = os.environ['OPENAI_APIKEY']


def run_customer_support(query: str) -> str:
    print(f"🔍 Received Query: {query}")  # Debug: Check if function is called

    try:
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        print("✅ OpenAI LLM Initialized")  # Debug: Check if OpenAI is loading

        agent = initialize_agent(tools=[],
                                 llm=llm,
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=True)
        print("🤖 LangChain Agent Initialized")  # Debug: Confirm agent setup

        response = agent.run(query)
        print(
            f"📝 Agent Response: {response}")  # Debug: Check if response exists

        return response

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")  # Debug: Print errors
        return "An error occurred while processing your request."
