import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import ollama
import uuid

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

pdf_path = "/Users/apple/Desktop/advisor/documents/english.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(pdf_text)
documents = [Document(page_content=chunk) for chunk in chunks]

collection_name = "constitution_chunks"
qdrant = QdrantClient(":memory:")  

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

qdrant_db = Qdrant.from_documents(
    documents=documents,
    embedding=embedding_model,
    qdrant_client=qdrant,
    collection_name=collection_name,
)

