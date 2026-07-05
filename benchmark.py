import json
import numpy as np
import os
import sys
from app import RAGPipeline, GeminiJudge
from document_processor import HierarchialProcessor
from vector_store import l2_normalize

# 1. The Golden Dataset
GOLDEN_DATASET = [
    {
        "question": "What was the total Segment Operating Income for the Intelligent Cloud segment in 2025 compared to 2024?",
        "expected_answer": "Intelligent Cloud operating income was $44,589 million in 2025 and $37,813 million in 2024.",
        "expected_parent_id": "page_38" # We check if the retriever found the right page!
    },
    {
        "question": "What was the total Research and development expense for 2025, and what drove the increase?",
        "expected_answer": "Research and development expense was $32,488 million in 2025. The 10% increase was driven by investments in cloud and AI engineering and Gaming.",
        "expected_parent_id": "page_40" 
    },
    {
        "question": "Looking at the Contractual Obligations table, what is the total amount due in 2026 for Purchase commitments?",
        "expected_answer": "The total amount due in 2026 for Purchase commitments is $103,940 million.",
        "expected_parent_id": "page_43" 
    }
]

print("Initializing Evaluation Pipeline...")
pipeline = RAGPipeline()
judge = GeminiJudge()
processor = HierarchialProcessor()

print("Parsing PDF...")
pdf_path = "temp_upload.pdf"
if not os.path.exists(pdf_path):
    print("Error: temp_upload.pdf not found. Please upload a PDF in the Streamlit app first.")
    sys.exit(1)

nodes = processor.process_pdf(pdf_path)

# Embed and store text nodes
print("Embedding text nodes...")
text_nodes = [n for n in nodes if n["type"] == "text"]
embeddings = pipeline.embedder.batch_embeddings([n["content"] for n in text_nodes])

emb_idx = 0
for node in nodes:
    if node["type"] == "text":
        norm_emb = l2_normalize(np.array(embeddings[emb_idx])).tolist()
        pipeline.store.add_node(node, norm_emb)
        emb_idx += 1
    else:
        pipeline.store.add_node(node)

pipeline.store.embedding_matrix = np.array([
    pipeline.store.nodes[node_id]["embedding"] 
    for node_id in pipeline.store.ordered_node_ids
])

print("Vector Store is fully populated!")

# 2. Evaluation Loop
total_mrr = 0
total_recall = 0
total_faithfulness = 0
total_relevance = 0

for item in GOLDEN_DATASET:
    query = item["question"]
    expected_parent_id = item["expected_parent_id"]
    expected_answer = item["expected_answer"]
    
    print(f"\n--- Testing: {query} ---")
    
    # Run the pipeline
    embedd = pipeline.embedder.get_embedding(query)
    results = pipeline.store.search_hybrid(query, np.array(embedd), top_k=5)
    
    # Calculate MRR and Recall
    rank = 0
    found = False
    for i, (child_node, score) in enumerate(results):
        if child_node["parent_id"] == expected_parent_id:
            rank = i + 1
            found = True
            break
            
    mrr = 1.0 / rank if found else 0.0
    recall = 1.0 if found else 0.0
    
    print(f"MRR: {mrr:.2f}, Recall: {recall}")
    total_mrr += mrr
    total_recall += recall
    
    # Generate the actual answer
    actual_answer = pipeline.answer_question(query)
    print(f"Model Answer: {actual_answer}")
    
    # Call the Judge
    judge_prompt = f"""
    You are an expert AI judge. Evaluate the following RAG system output.
    
    Question: {query}
    Expected Answer: {expected_answer}
    Actual Answer: {actual_answer}
    
    Score 'faithfulness' (1-5): Does the Actual Answer rely strictly on the context and align with the Expected Answer without hallucinating?
    Score 'relevance' (1-5): Does the Actual Answer directly address the Question?
    
    Return a strict JSON object:
    {{"faithfulness": <int>, "relevance": <int>}}
    """
    
    judge_response = judge.evaluate(judge_prompt)
    try:
        scores = json.loads(judge_response)
        faithfulness = scores.get("faithfulness", 0)
        relevance = scores.get("relevance", 0)
    except Exception as e:
        print("Judge returned invalid JSON.", judge_response)
        faithfulness = 0
        relevance = 0
        
    print(f"Judge Scores -> Faithfulness: {faithfulness}/5, Relevance: {relevance}/5")
    total_faithfulness += faithfulness
    total_relevance += relevance

n = len(GOLDEN_DATASET)
print("\n=== FINAL BENCHMARK REPORT ===")
print(f"Mean Reciprocal Rank (MRR): {total_mrr/n:.2f}")
print(f"Context Recall: {total_recall/n:.2f}")
print(f"Average Faithfulness: {total_faithfulness/n:.2f}/5.0")
print(f"Average Relevance: {total_relevance/n:.2f}/5.0")
