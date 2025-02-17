from fastapi import FastAPI
import uvicorn
from Agents.customer_support import run_customer_support

app = FastAPI()


@app.get("/")
def read_root():
  return {"message": "Multi-Agent Business Automation System is running."}


@app.get("/customer-support")
def customer_support(q: str = Query("Hello, I need help!",
                                    description="User query")):
  """
    API endpoint for the Customer Support Agent.
    Example: /customer-support?q=I have a problem with my order
    """
  response = run_customer_support(q)
  return {"response": response}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
