from Agents.customer_support import run_customer_support

test_query = "Where is my order?"
response = run_customer_support(test_query)
print(f"🤖 Bot Response: {response}")
