import streamlit as st
import numpy as np
from app import RAGPipeline
from document_processor import split_text
from vector_store import l2_normalize

# 1. Initialize our RAG Pipeline in Streamlit's session state
# (This ensures it doesn't get reset every time the user clicks a button)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.is_ready = False

st.title(" Medical AI Assistant")
st.markdown("Upload a medical text document and ask questions about it!")

# 2. File Uploader Widget
uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

if uploaded_file is not None and not st.session_state.is_ready:
    with st.spinner("Processing document..."):
        # Read the raw text from the uploaded file
        raw_text = uploaded_file.getvalue().decode("utf-8")
        
        chunks = split_text(raw_text, chunk_size=200, overlap=50) 
        chunk_texts = [c["text"] for c in chunks]

        # 1. Embed and Store
        embeddings = st.session_state.pipeline.embedder.batch_embeddings(chunk_texts)
        for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
            norm_emb = l2_normalize(np.array(emb)).tolist()
            st.session_state.pipeline.store.add_document(text, norm_emb, doc_id=i)
        
        # 2. Build the Matrix
        st.session_state.pipeline.store.embedding_matrix = np.array([doc["embedding"] for doc in st.session_state.pipeline.store.documents])
    
        st.session_state.is_ready = True
        st.success("Document processed and loaded into Vector Database!")

# 3. Chat Interface
if st.session_state.is_ready:
    user_question = st.chat_input("Ask a question about the document...")
    
    if user_question:
        st.chat_message("user").write(user_question)
        
        # 4. Generate Answer
        answer = st.session_state.pipeline.answer_question(user_question)
        
        st.chat_message("assistant").write(answer)