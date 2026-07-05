import contextlib
import numpy as np
from vector_store import SimpleVectorStore, EmbeddingModel
import google.generativeai as genai
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate(self, prompt: list) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class GeminiJudge:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # We enforce a strict JSON output schema!
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        
    def evaluate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class RAGPipeline:
    def __init__(self):
        self.store = SimpleVectorStore()
        self.embedder = EmbeddingModel()
        self.llm = GeminiLLM()

    def answer_question(self, user_query: str) -> str:
        embedd = self.embedder.get_embedding(user_query)
        results = self.store.search_hybrid(user_query, np.array(embedd), top_k=2)
        context_pieces = []

        image_pieces = []
        seen_parents = set()

        for child_node, score in results:
            parent_id = child_node["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_node = self.store.nodes[parent_id]
                context_pieces.append(parent_node["content"])
                for node in self.store.nodes.values():
                    if node.get("parent_id") == parent_id and node["type"] == "image":
                        image_pieces.append(node["content"])
        
        full_context = "\n".join(context_pieces)
        prompt = f"""
         you are an intelligent AI assistant.Anwer logically based ONLY on the provided context.
           Context: {full_context}
           Question: {user_query}
           """
        final_payload = [prompt] + image_pieces
        return self.llm.generate(final_payload)

            
if __name__ == "__main__":
    from vector_store import l2_normalize
    pipeline = RAGPipeline()
    print("Loading models...")
    
    # 1. Load some dummy data
    docs = [
        "Myocardial infarction is commonly known as a heart attack. Symptoms include severe chest pain.",
        "Patient PT-9942X was admitted to the ER for a severe asthma attack.",
        "A healthy diet and exercise can prevent many cardiovascular diseases."
    ]
    
    # 2. Embed and add to Vector Store
    print("Embedding documents...")
    embeddings = pipeline.embedder.batch_embeddings(docs)
    for i, (text, emb) in enumerate(zip(docs, embeddings)):
        norm_emb = l2_normalize(np.array(emb)).tolist()
        pipeline.store.add_document(text, norm_emb, doc_id=i)
        
    pipeline.store.embedding_matrix = np.array([doc["embedding"] for doc in pipeline.store.documents])
    
    # 3. Ask a question!
    query = "Did patient PT-9942X have a heart attack?"
    print(f"\nUser Query: {query}")
    
    response = pipeline.answer_question(query)
    print(response)
