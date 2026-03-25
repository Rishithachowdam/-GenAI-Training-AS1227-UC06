from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import AzureOpenAI
from dotenv import load_dotenv
from llm_wrapper import AzureLLM
import os

load_dotenv()

# Azure Config
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# ✅ Azure Client (FIRST)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ✅ LLM Wrapper (AFTER client)
llm = AzureLLM(client, DEPLOYMENT_NAME)

# 🔹 Embeddings (HuggingFace)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# 🔹 Create Vector DB
def create_vectorstore(chunks):
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return db

# 🔹 Load DB
def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

# 🔹 Retrieve Docs
def retrieve_docs(query, db):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever.invoke(query)

# 🔹 Generate Answer (UPDATED)
def generate_answer(query, docs):
    context = ""
    sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        path = doc.metadata.get("file_path", "")

        if path:
            sources.add(path)

        context += f"""
Document: {source}
Page: {page}
Path: {path}

Content:
{doc.page_content}
---------------------
"""

    # 🔗 Prepare unique links
    links_text = "\n".join([f"- {s}" for s in sources])

    prompt = f"""
You are an Employee Onboarding Assistant.

STRICT RULES:
- Answer ONLY from the provided context
- DO NOT use outside knowledge
- ALWAYS include document links ONLY from the "Available Documents" section
- If multiple documents are relevant, include ALL corresponding links
- DO NOT generate, modify, or guess any links
- If answer is not found → respond exactly with: "I don’t know based on the available documents"

- Always provide the answer in bullet points format as shown below:
  Example:
    • Yes, probation can be extended if performance expectations are not fully met.
    • The extension period can be up to 2 months.
    • The decision must be communicated formally.

- Always include document name and page/section in "Policy Reference" if available:
  Example:
    • Document: Probation Confirmation Policy
    • Section/Page: Page 4

- If multiple documents are used:
    • Mention ALL document names and corresponding pages

- If the question is basic conversation (e.g., "Hi", "Hello", "Hey"):
    • Respond conversationally
    • DO NOT follow the structured format
    • DO NOT include document links
- Maintain proper indentation between headings, subheadings, and bullet points in the answer
- Keep answers clear, concise, and directly relevant to the question
- Do not repeat the question
- Do not include unnecessary explanations


Context:
{context}

Available Documents:
{links_text}

Question:
{query}

Answer format:
• Answer:
• Policy Reference:
• Document Link: (must include only links from Available Documents)
• Next Steps:
• Escalation/Contact:
"""

    # ✅ ONLY ONE LLM CALL (Correct)
    answer, latency, pt, ct, tt = llm.invoke(prompt)

    return answer, latency, pt, ct, tt