from fastapi import FastAPI, Query
import sys
import os
import uvicorn

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Agents.customer_support import run_customer_support

app = FastAPI()


@app.get("/")
def read_root():
  print("✅ FastAPI is running!")
  sys.stdout.flush()
  return {"message": "Multi-Agent Business Automation System is running."}


@app.get("/customer-support")
def customer_support(q: str = Query("Hello, I need help!",
                                    description="User query")):
  """
    API endpoint for the Customer Support Agent.
    """
  print(f"🔍 FastAPI Received Query: {q}")  # Debugging: Confirm API call
  sys.stdout.flush()

  response = run_customer_support(q)

  print(
      f"📨 Returning API Response: {response}")  # Debugging: Check final output
  sys.stdout.flush()

  return {"response": response}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
