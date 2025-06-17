from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Load and split the resume
pdf_loader = PyPDFLoader("data.pdf")
pages = pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_text("\n\n".join(p.page_content for p in pages))

# Generate embeddings and save
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

Chroma.from_texts(
    texts,
    embedding=embeddings,
    persist_directory="chroma_db"
).persist()

print("✅ Chroma vector DB built and saved to 'chroma_db/'")
