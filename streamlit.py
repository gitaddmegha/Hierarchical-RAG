import streamlit as st
import numpy as np
from app import RAGPipeline
from document_processor import HierarchialProcessor
from vector_store import l2_normalize

# 1. Initialize our RAG Pipeline in Streamlit's session state
# (This ensures it doesn't get reset every time the user clicks a button)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.is_ready = False

st.title(" Medical AI Assistant")
st.markdown("Upload a medical text document and ask questions about it!")

# 2. File Uploader Widget
uploaded_file = st.file_uploader("Upload a Financial Report (.pdf)", type = ["pdf"])
if uploaded_file is not None and not st.session_state.is_ready:
    with st.spinner("Parsing Hierarchical Graph and Extracting Images..."):
        temp_path = "temp_upload.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processor = HierarchialProcessor()
        nodes = processor.process_pdf(temp_path)
         #embed and store

        text_nodes = [n for n in nodes if n["type"] == "text"]
        embeddings = st.session_state.pipeline.embedder.batch_embeddings([n["content"] for n in text_nodes])
        emb_idx = 0
        for node in nodes:
            if node["type"] == "text":
                norm_emb = l2_normalize(np.array(embeddings[emb_idx])).tolist()
                st.session_state.pipeline.store.add_node(node, norm_emb)
                emb_idx += 1
            else:
                st.session_state.pipeline.store.add_node(node)
        st.session_state.pipeline.store.embedding_matrix = np.array([
            st.session_state.pipeline.store.nodes[node_id]["embedding"] 
            for node_id in st.session_state.pipeline.store.ordered_node_ids
        ])
    
        st.session_state.is_ready = True
        st.success("Graph constructed! Ready for FinQA.")



# 3. Chat Interface
if st.session_state.is_ready:
    user_question = st.chat_input("Ask a question about the document...")
    
    if user_question:
        st.chat_message("user").write(user_question)
        
        # 4. Generate Answer
        answer = st.session_state.pipeline.answer_question(user_question)
        
        st.chat_message("assistant").write(answer)