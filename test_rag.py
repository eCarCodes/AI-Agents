from langchain_ollama import OllamaLLM

# Initialize the local brain
model = OllamaLLM(model="llama3.1")

# Ask a simple question
response = model.invoke("Who is the current president of the United States of America?")
print(response)