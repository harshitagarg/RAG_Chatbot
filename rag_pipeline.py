# Import Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")

# Import necessary libraries
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_retriever(query):
    query = query.lower()
    
    if "sql" in query:
        return sql_retriever
    elif "rag" in query or "llm" in query:
        return rag_retriever
    else:
        return llm_retriever  # default
    
# Load the PDF
folder_path = "data/"

all_documents = []


for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        docs = loader.load()

        for doc in docs:
            doc.metadata = {
                "source": file.lower(),              # 👈 clean file name
                "page": doc.metadata.get("page", None)
            }

        all_documents.extend(docs)

# Split documents into chunks of 500 characters with 100 characters overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(all_documents)


# print(all_documents)
# Create embeddings for the text chunks
embedding = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs = {'trust_remote_code': True})

sql_docs = [doc for doc in texts if "sql" in doc.metadata["source"]]
rag_docs = [doc for doc in texts if "rag" in doc.metadata["source"]]
llm_docs = [doc for doc in texts if "llm" in doc.metadata["source"]]

sql_retriever = FAISS.from_documents(sql_docs, embedding).as_retriever(search_type='mmr',search_kwargs={"k": 5, "fetch_k":10})
rag_retriever = FAISS.from_documents(rag_docs, embedding).as_retriever(search_type='mmr',search_kwargs={"k": 5, "fetch_k":10})
llm_retriever = FAISS.from_documents(llm_docs, embedding).as_retriever(search_type='mmr',search_kwargs={"k": 5, "fetch_k":10}) 

from langchain.prompts import PromptTemplate

prompt_template = """

You are an AI assistant.

Use the context to answer factual questions.

If the question is asking for:
- examples
- interview questions
- suggestions
- explanations

Then you are allowed to generate answers based on your knowledge,
while using the context as guidance.

If the answer is completely unrelated to context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Import the RetrievalQA chain for question-answering tasks
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Define a query and retrieve relevant documents
# query = "Please generate 5 sql queries with answer"

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.9
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

print("🤖 Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    retriever = get_retriever(query)

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    response = chat_chain.invoke({"question": query})

    print("\nBot:", response["answer"], "\n")
