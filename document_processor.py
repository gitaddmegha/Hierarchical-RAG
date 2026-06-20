"""

Handles reading text files and chunking them using a sliding window approach.
"""

from typing import List, Dict

def read_file(file_path: str) -> str:
    with open(file_path, "r",encoding ="UTF-8") as file:
        return file.read()

def split_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, any]]:
    """
    Splits a string into overlapping chunks.
    
    Args:
        text (str): The raw input text.
        chunk_size (int): Max character length of a chunk.
        overlap (int): Number of characters to overlap between chunks.
        
    Returns:
        List[Dict[str, any]]: A list of dictionaries representing chunks.
    """

    if chunk_size <= 0 or overlap < 0:
        raise ValueError("Chunk size must be positive and overlap must be non-negative.")
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    chunk_id = 0
    step_size = chunk_size - overlap
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start: end]
        chunk_data = {"id" : chunk_id, "text" : chunk_text,
        "start_char" : start, "end_char" : end
        }
        chunks.append(chunk_data)
        start += step_size
        chunk_id += 1
    return chunks

    pass
