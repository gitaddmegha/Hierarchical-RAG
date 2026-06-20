import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import split_text

def run_tests():
    # Test 1: Standard Sliding Window
    text1 = "I am Megha"
    chunks1 = split_text(text1, chunk_size=5, overlap=2)
    print("Test 1 (Standard):")
    for c in chunks1:
        print(c)
    print("-" * 40)

    # Test 2: Short Text (less than chunk_size)
    text2 = "Hi"
    chunks2 = split_text(text2, chunk_size=5, overlap=2)
    print("Test 2 (Short Text):")
    for c in chunks2:
        print(c)
    print("-" * 40)

    # Test 3: Empty Text
    text3 = ""
    chunks3 = split_text(text3, chunk_size=5, overlap=2)
    print("Test 3 (Empty Text):")
    for c in chunks3:
        print(c)
    print("-" * 40)

    # Test 4: Validation Guard Check (Should raise ValueError)
    print("Test 4 (Validation Guard):")
    try:
        split_text("validation test", chunk_size=5, overlap=5)
        print("FAIL: Validation did not raise ValueError!")
    except ValueError as e:
        print(f"SUCCESS: Caught expected exception: {e}")
    print("-" * 40)

if __name__ == "__main__":
    run_tests()
