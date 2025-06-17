from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS  # more stable than Chroma
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pypdf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://rishabhpandey-kappa.vercel.app",
    "https://www.rizzuppandey.me"
]}})
load_dotenv()

pdf_loader = PyPDFLoader("data.pdf")
pages = pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_text("\n\n".join([p.page_content for p in pages]))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_index = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

QA_CHAIN_PROMPT = PromptTemplate.from_template("""
You are the candidate described in the resume below. Your job is to answer questions strictly based on this resume context only.
Never attempt to answer anything outside of the resume, especially general knowledge or world-related questions.
If the answer is not in the resume, respond creatively or humorously to indicate that you don't know — but never fabricate.

Be brief, relevant, and resume-focused.

Resume:
{context}

Question: {question}

Answer:
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

@app.route("/health", methods=["GET"])
def health():
    return "App is healthy!", 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"answer": result["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
