from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# STEP 1: Load the cleaned book
file_path = "three_pieces_of_os_cleaned.txt"
if not os.path.exists(file_path):
    print("❌ Cleaned text file not found.")
    exit()

with open(file_path, "r", encoding="utf-8") as f:
    book_text = f.read()

# STEP 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([book_text])
print(f"✅ Book split into {len(docs)} chunks.")

# STEP 3: Create and store embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)
db.save_local("os_vector_store")
print("✅ Vector store saved.")

# STEP 4: Load local LLM (or switch to OpenAI if needed)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)
print("✅ LLM loaded.")

# STEP 5: Setup QA system
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# STEP 6: Ask questions
print("\n📘 Ask any question about 'Three Pieces of OS'. Type 'exit' to quit.")
while True:
    query = input("\n❓ Question: ")
    if query.lower() == 'exit':
        print("👋 Exiting...")
        break
    response = qa.run(query)
    print("\n📘 Answer:", response)
