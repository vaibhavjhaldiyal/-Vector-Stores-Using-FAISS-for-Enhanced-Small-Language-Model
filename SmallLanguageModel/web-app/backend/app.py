from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "os_vector_store")

# Create Flask app
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# Load vector store and model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}

Answer:"""
)

# QA chain with custom prompt
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please provide a question."})
    answer = qa.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

