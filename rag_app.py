from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- 1. CONFIGURATION ---
FILE_PATH = "your_document.pdf"  # Put your PDF filename here!
DB_DIR = "./chroma_db"

# --- 2. THE "KNOWLEDGE" PART (Ingestion) ---
print("📚 Loading and splitting document...")
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()

# Chop the text into pieces so the AI can find specific sections
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Create the "Library" (Vector DB)
print("🧠 Creating vector database (this might take a moment)...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=DB_DIR
)

# --- 3. THE "CHATTING" PART (Retrieval) ---
llm = OllamaLLM(model="llama3.1")

# Create a "Chain" that knows how to:
# 1. Search the DB for relevant chunks
# 2. Feed them to Llama 3.1
# 3. Give you the answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "Stuff" just means "stuff all the found info into the prompt"
    retriever=vector_db.as_retriever()
)

# --- 4. ASK A QUESTION ---
print("\n✅ Setup complete! Ask your document a question.")
query = "What is the main topic of this document?"
response = qa_chain.invoke(query)

print(f"\nQuestion: {query}")
print(f"Answer: {response['result']}")