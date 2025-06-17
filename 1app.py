from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# Allow frontends
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://rishabhpandey-kappa.vercel.app",
            "https://www.rizzuppandey.me"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": "*"
    }
})

# Load prebuilt vector index
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_index = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
).as_retriever(search_kwargs={"k": 3})

# Gemini model
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
    convert_system_message_to_human=True
)

# Prompt for strict resume-only answers
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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = qa_chain({"query": question})
        return jsonify({"answer": result["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
