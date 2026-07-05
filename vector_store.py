"""
Custom Vector Database and Embeddings from Scratch with Hybrid Search.
"""
import document_processor
import math
import string
import json
import numpy as np
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector) 
    return vector / norm if norm > 0 else vector

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Length of both vectors must be same")

    dot_prod = sum( a * b for a,b in zip(v1,v2))
    v1_mag = math.sqrt(sum(a*a for a in v1))
    v2_mag = math.sqrt(sum(b*b for b in v2))

    if v1_mag == 0.0 or v2_mag == 0.0:
        return 0.0
    return dot_prod / (v1_mag * v2_mag)

def tokenize(text: str)-> List[str]:
    text_lower = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text_lower.translate(translator)

    return clean_text.split()


class EmbeddingModel:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self,text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy = True)
        return embedding.tolist()

    def batch_embeddings(self, texts: List[str])-> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts,convert_to_numpy = True)
        return embeddings.tolist()


class SimpleVectorStore:
    def __init__(self):
        self.nodes: Dict[str,Any] = {}
        self.embedding_matrix: np.ndarray = None
        self.ordered_node_ids: List[str] = []
    def add_node(self, node: Dict, embedding: List[float] = None) -> None:
         """Adds a node to the graph. Only child text nodes receive embeddings."""
         if embedding is not None:
            node["embedding"] = embedding
            self.ordered_node_ids.append(node["id"])
         self.nodes[node["id"]] = node

    
    def save(self, file_path : str) -> None:
        with open(file_path, "w", encoding = "UTF-8")as f:
            json.dump(self.documents, f, indent = 2)
    
    def load(self, file_path: str) -> None:
        with open(file_path, "r", encoding="UTF-8") as f:
            self.documents = json.load(f)


    def search_dense(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str,Any], float]]:
        """
        Performs fully vectorized similarity search using matrix-vector multiplication.
        Assumes self.embedding_matrix is a 2D array and query_embedding is a 1D array.
        """
        if self.embedding_matrix is None or len(self.ordered_node_ids) == 0:
            return []
    
        norm = l2_normalize(query_embedding)
        score = np.dot(self.embedding_matrix, norm)
        sorted_idx = np.argsort(score)[::-1]
        results = []
        for i in sorted_idx[:top_k]:
            node_id = self.ordered_node_ids[i]
            node = self.nodes[node_id]
            results.append((node, score[i])) 
        return results


    def search_keyword(self, query_text: str, top_k: int= 3) -> List[Tuple[Dict[str,Any], float]]:
        text_nodes = [n for n in self.nodes.values() if n["type"] == "text"]
        if not text_nodes:
            return []
        
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return [(doc, 0.0) for doc in text_nodes[:top_k]]
        
        N = len(text_nodes)
        df = {}
        for token in query_tokens:
            count = sum(1 for doc in text_nodes if token in tokenize(doc["content"]))
            df[token] = count

        idf = {}
        for token in query_tokens:
            df_t = df[token]
            if df_t > 0:
                idf[token] = math.log(1.0 + N/df_t)
            else:
                idf[token] = 0.0
            
        scored_docs = []
        for doc in text_nodes:
            doc_tokens = tokenize(doc["content"])
            doc_len = len(doc_tokens)
            score = 0.0
            if doc_len > 0:
                for token in query_tokens:
                    tf = doc_tokens.count(token) / doc_len
                    score += tf * idf[token]
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x:x[1], reverse = True)
        return scored_docs[:top_k]

    def search_hybrid(self, query_text: str, query_embedding: np.ndarray, top_k: int=3, alpha: float= 0.5) -> List[Tuple[Dict[str, Any], float]]:
        dense_results = self.search_dense(query_embedding, top_k = 10)
        keyword_results = self.search_keyword(query_text, top_k = 10)

        max_dense = dense_results[0][1] if dense_results else 1e-9
        max_keyword = keyword_results[0][1] if keyword_results else 1e-9

        combined = {}
        for doc, score in dense_results:
            doc_id = doc["id"]
            if doc_id not in combined:
                combined[doc_id] = {"doc": doc, "score": 0.0}
            combined[doc_id]["score"] += alpha * score/max_dense
        for doc, score in keyword_results:
            doc_id = doc["id"]
            if doc_id not in combined:
                combined[doc_id] = {"doc": doc, "score": 0.0}
            combined[doc_id]["score"] += (1-alpha) * score/max_keyword
        final_results = [(data["doc"], data["score"]) for data in combined.values()]
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]