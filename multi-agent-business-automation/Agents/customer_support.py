import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

# Retrieve the API key from Replit Secrets
OPENAI_API_KEY = os.environ['OPENAI_APIKEY']


def run_customer_support(query: str) -> str:
    """
    A simple function that uses LangChain to process a customer support query.
    """
    # Pass the API key to OpenAI
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    # Initialize a minimal agent without additional tools (yet).
    agent = initialize_agent(tools=[],
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    # Process the incoming query and return the response
    response = agent.run(query)
    return response
