from ingest import load_documents, split_documents
from rag_pipeline import create_vectorstore

# Step 1: Load documents
docs = load_documents("data")
print(f"Loaded {len(docs)} pages")

# Step 2: Split into chunks
chunks = split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Create vector DB
create_vectorstore(chunks)

print("✅ Vector DB created successfully!")