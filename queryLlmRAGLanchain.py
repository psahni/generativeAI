# pip install langchain qdrant-client transformers

# Connect to Vector DB
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(host="localhost", port=6333)

vector_store = Qdrant(
    client=client,
    collection_name="research_chunks",
    embeddings=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Setup Your Custom Mistral LLM in LangChain

from langchain.llms.base import LLM
import requests

class CustomMistralLLM(LLM):
    endpoint_url: str

    def _call(self, prompt, stop=None):
        response = requests.post(self.endpoint_url, json={"prompt": prompt})
        return response.json()["response"]

    @property
    def _llm_type(self):
        return "custom-mistral"

llm = CustomMistralLLM(endpoint_url="http://your-host:8000/generate")

# Build Retrieval-Augmented QA Chain

from langchain.chains import RetrievalQA

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # Stuff, MapReduce, Refine
    return_source_documents=True
)

# Query

query = "What is the outcome of clinical trial for device X?"
result = rag_chain(query)

print("Answer:", result['result'])
print("Sources:", [doc.metadata['source'] for doc in result['source_documents']])

