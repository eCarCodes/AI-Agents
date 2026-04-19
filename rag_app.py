from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

FILE_PATH = "./data/location.txt" # Desired .txt file
DB_DIR = "./chroma_db"
# Ingestion (knowledge) of file/s
print("Loading & Splitting...")
loader = TextLoader(FILE_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250,
    separators=["\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(docs)
# The "Library" of the RAG process
print("Creating Vector Database...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)
# Import desired model 
llm = OllamaLLM(model="llama3.1")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)
# Ask the bot a test question to verify successful file retrieval
query = "What is the main topic of this imported file?"
response = qa_chain.invoke(query)
print(f"\nQuestion: {query}")
print(f"Answer: {response['result']}")