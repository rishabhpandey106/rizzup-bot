from flask import Flask, request, jsonify
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from flask_cors import CORS
import pypdf


app = Flask(__name__)
load_dotenv()
cors = CORS(app, resources={r"/*": {"origins": "https://rishabhpandey-kappa.vercel.app"}})
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pdf_loader = PyPDFLoader("data1.pdf")
pages = pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.2, convert_system_message_to_human=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        QA_CHAIN_PROMPT = PromptTemplate.from_template(
            """You are the candidate (mentioned in your resume) answering questions based on your resume. Use the following pieces of context as your resume to answer the question at the end. If you don't know the answer don't ever mention something like "my resume doesn't include" in answer,  but tell them you don't know in humorous/creative way. Keep the answer as concise as possible.
            {context}
            Question: {question}
            Helpful Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa_chain({"query": question})
        return jsonify({"answer": result["result"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
