from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# 1. Dummy documents
documents = [
    Document(page_content="Article 21 guarantees the right to life and personal liberty."),
    Document(page_content="Article 19 ensures freedom of speech and expression."),
    Document(page_content="Article 370 granted special status to Jammu and Kashmir."),
]

# 2. Split (optional here)
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
docs = splitter.split_documents(documents)

# 3. Use sentence-transformers embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embedding_model)

# 5. Query the RAG system
query = "What does Article 21 talk about?"
result = vectorstore.similarity_search(query, k=2)

# 6. Show result
for doc in result:
    print("🔹", doc.page_content)
