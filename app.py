import os
import streamlit as st
import time   
import pandas as pd  
from rag_pipeline import load_vectorstore, retrieve_docs, generate_answer


# Auto-create vectorstore if missing
if not os.path.exists("vectorstore"):
    from ingest_runner import load_documents, split_documents, create_vectorstore
    
    st.write(" Initializing knowledge base... Please wait ")
    docs = load_documents("data")
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    st.write(" Knowledge base ready!")

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Onboarding Assistant", page_icon="logo.png", layout="wide")

# ------------------------
#  CUSTOM CSS (UI MAGIC)
# ------------------------
st.markdown("""
<style>

/*  Reduce top spacing */
.block-container {
    padding-top: 2rem;
}

/*  HEADER CARD */
.custom-header {
    position: Sticky;
    max-width: 900px;
    margin: 0 auto; /* center */
    background: linear-gradient(90deg, #2b6cb0, #38a169);
    padding: 18px 30px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/*  HEADER TEXT */
.header-text {
    font-size: 26px;
    font-weight: bold;
    color: white;
    text-align: center;
}

/*  TAGLINE */
.tagline {
    position: Sticky;
    text-align: center;
    font-size: 14px;
    color: #666;
    margin-top: 8px;
    margin-bottom: 25px;
}

/*  SIDEBAR FIX */
section[data-testid="stSidebar"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

/*  LOGO ADJUST */
.logo-container img {
    margin-top: -15px !important;
}

/*  ABOUT TEXT */
.about-text {
    font-size: 12px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
#  HEADER SECTION
# ------------------------
st.markdown("""
<div class="custom-header">
    <div class="header-text">🤖 Employee Onboarding Assistant</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tagline">
Ask anything about company policies, onboarding steps, and processes.</div>
""", unsafe_allow_html=True)

# ------------------------
# Session State (Chat History)
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("logo.png", width=180)
    st.markdown('</div>', unsafe_allow_html=True)

    st.header("⚙️ Controls")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**About:**")
    st.markdown('<div class="about-text">This assistant answers based only on company documents.</div>', unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ------------------------
# Display Chat History
# ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------
# User Input
# ------------------------
query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    db = load_vectorstore()

    with st.spinner("Thinking... 🤔"):

        #  Measure Retrieval Time
        t1 = time.time()
        docs = retrieve_docs(query, db)
        retrieval_time = time.time() - t1

        #  LLM Call (UPDATED RETURN VALUES)
        answer, llm_latency, pt, ct, tt = generate_answer(query, docs)

        total_time = retrieval_time + llm_latency

        #  Cost Calculation (EDIT if needed based on your model)
        input_cost = (pt / 1000) * 0.01
        output_cost = (ct / 1000) * 0.03
        total_cost = input_cost + output_cost

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        #  METRICS DISPLAY
        st.markdown(f"""
---
### Metrics
 **Latency**
- Retrieval: {retrieval_time:.2f}s  
- LLM: {llm_latency:.2f}s  
- Total: {total_time:.2f}s  

 **Cost**
- Prompt Tokens: {pt}  
- Completion Tokens: {ct}  
- Total Tokens: {tt}  
- Cost: ${total_cost:.6f}
""")

    #  LOGGING
    log = pd.DataFrame([{
        "query": query,
        "retrieval_time": retrieval_time,
        "llm_latency": llm_latency,
        "total_time": total_time,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "cost": total_cost
    }])

    log.to_csv("metrics_log.csv", mode="a", header=not os.path.exists("metrics_log.csv"), index=False)

st.markdown('</div>', unsafe_allow_html=True)