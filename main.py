# import Libraries and Load Models
from langchain_community.llms.ctransformers import CTransformers
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
import faiss
import torch
import numpy as np

# Load Models
llm = CTransformers(
    model="TheBloke/Llama-2-7b-GGML",
    model_type="llama"
)
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Load model
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

#Get Embeddings Dimentions
embeddings_exmple = embed_texts("Hello, how are you?")
embeding_dim = embeddings_exmple.shape[1]

#initialize FAISS index 
index = faiss.IndexFlatL2(embeding_dim)

# Initialize InMemoryDocstore
docstore = InMemoryDocstore()

#Create an index-to-document mapping
index_to_docstore_id = {}

#create the fiass vector store
vector_store = FAISS(embedding_function=embed_texts, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

#prepare documents
documents = [
    Document(page_content="RAG (Retrieval-Augmented Generation) is a method that combines a language model with an external database or documents, so the model can fetch relevant information before generating an answer."),
    Document(page_content="RAG is commonly used in chatbots, question-answering systems, and search-based AI apps, because it reduces hallucination and improves reliability."),
    Document(page_content="This approach helps the model produce more accurate, updated, and factual responses, especially when the needed information is not inside the model itself."),
]

#embed documents and add to the vector store
texts = [doc.page_content for doc in documents]
embeddings = get_embeddings(texts)

for i, embedding in enumerate(embeddings):
    index.add(np.array([embedding], dtype=np.float32))
    index_to_docstore_id[i] = documents[i].page_content

#Define a simple retriever
def simple_retriever(query):
  query_embedding = embed_texts([query])
   D, I = index.search(query_embedding, k=1)
    return index_to_docstore_id[I[0][0]] if len(I) > 0 and I[0][0] in index_to_docstore_id else None

#Create the RAG Chain
class SimpleRetrieverlQA:
    def __init__(self, retriever):
        self.llm = llm
        self.retriever = retriever

    def run(self, query):
        return self.retriever(query)
        response = self.llm(f"Context: {context}\nQuestion: {query}")
  return response

qa_chain = SimpleRetrieverlQA(llm = llm, retriever = simple_retriever)

#Questions
questions = ["What is RAG?", "Why is RAG used?", "How does RAG work?"]  

#Get Answers
answers = qa_chain.run(questions)
print(answers)