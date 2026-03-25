# Overview
- This project is an AI-powered Employee Onboarding Assistant designed to help employees quickly access and understand company policies.
- Instead of manually searching through multiple documents, users can ask questions in natural language and receive:
- Accurate answers based only on company documents
- Policy references (document name + page)
- Direct links to source documents

# Approach
This solution is built using a RAG (Retrieval-Augmented Generation) pipeline:
- 1.Documents are ingested and converted into text
- 2.Text is split into smaller chunks
- 3.Each chunk is converted into embeddings
- 4.Embeddings are stored in a vector database (ChromaDB)
- 5.User queries are matched with relevant chunks
- 6.Retrieved context is passed to an LLM for answer generation

# RAG Architecture → Ensures answers are grounded in documents
- 1.Strict Prompt Engineering → Prevents hallucinations
- 2.Chunking Strategy (500 / 100 overlap) → Maintains context continuity
- 3.Metadata Tracking → Enables document references and traceability
- 4.Local Vector Store (ChromaDB) → Fast and cost-efficient retrieval
- 5.Streamlit UI → Simple and interactive chat interface
- 6.Metrics Logging → Tracks latency, tokens, and cost per query

# Tech Stack
- LLM: Azure OpenAI(GPT-4.1 Mini)
- Embeddings: HuggingFace (all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- Backend: Python
- Frontend: Streamlit

# How to Run the Project
- 1. Clone the Repository
   ```bash
    git clone <your-repo-link>
    cd Employee-Onboarding-Assistant

- 2. Install Dependencies
   ```bash
      pip install -r requirements.txt

- 3. Setup Environment Variables
        - Create a .env file in the root directory:
           - AZURE_OPENAI_KEY=your_api_key
           - AZURE_OPENAI_ENDPOINT=your_endpoint
           - AZURE_DEPLOYMENT_NAME=your_deployment_name

- 4.Add Documents
      - Place all policy documents inside the data/ folder.

- 5. Create Vector Database
   ```bash
       python ingest_runner.py

- 6. Run the Application
   ```bash
       streamlit run app.py
#App_Link: 
https://novasure-technologies-service.streamlit.app/
