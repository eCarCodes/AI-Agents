from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
# Connect the Chroma folder
DB_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")
vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
# Set up the brain (retrieval chain)
llm = OllamaLLM(model="llama3.1")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)
# Interactive loop
print("Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in {"exit", "quit", "goodbye", "bye"}:
        break
    response = qa_chain.invoke(query)
    print(f"\nAssistant: {response['result']}\n")
    print("-" * 40) # separator
print("Goodbye!")