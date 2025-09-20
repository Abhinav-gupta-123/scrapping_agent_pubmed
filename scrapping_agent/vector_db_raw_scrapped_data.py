import json
import math
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# -------------------------------
# Initialize embedding model
# -------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------
# Function 1: Build vector DB
# -------------------------------
def build_vector_db(json_path: str,
                    index_path: str = "my_faiss.index",
                    chunks_path: str = "chunks.json",
                    chunk_size: int = 300,
                    chunk_overlap: int = 50):
    """
    Load JSON data, chunk text, embed, and save FAISS index + chunks mapping
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_texts = []

    # -------------------------------
    # Extract text and chunk it
    # -------------------------------
    for entry in data:
        # Customize which field to embed
        text = entry.get("content", "")
        if not text.strip():
            continue
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            all_texts.append(chunk)
            start += chunk_size - chunk_overlap  # overlap for context

    print(f"Total chunks created: {len(all_texts)}")

    # -------------------------------
    # Generate embeddings
    # -------------------------------
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # -------------------------------
    # Build FAISS index
    # -------------------------------
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # Save chunks mapping
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, indent=2, ensure_ascii=False)

    print(f"✅ Vector DB built and saved: {index_path}, {chunks_path}")


# -------------------------------
# Function 2: Query vector DB
# -------------------------------
def query_vector_db(query: str,
                    index_path: str = "my_faiss.index",
                    chunks_path: str = "chunks.json",
                    top_k: int = 5) -> List[Dict]:
    """
    Load FAISS index + chunks mapping and search query
    """
    # Load index
    index = faiss.read_index(index_path)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Embed query
    query_vec = model.encode([query], convert_to_numpy=True)

    # Search
    distances, indices = index.search(query_vec, top_k)

    # Map results to original text
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": chunks[idx],
            "score": float(dist)
        })

    return results


# -------------------------------
# Usage Example
# -------------------------------
if __name__ == "__main__":
    # Step 1: Build vector DB
    build_vector_db("scraped_articles.json")

    # Step 2: Query vector DB
    query = "What is chronic cough? what are its common causes? what are the symptoms?"
    results = query_vector_db(query)

    print("\n🔍 Query Results:")
    if results:
        top_score = max(r['score'] for r in results)
        print(f"Top raw score: {top_score:.4f}")
        for i, r in enumerate(results, 1):
            normalized = (r['score'] / top_score) * 100 if top_score else 0
            print(f"{i}. [Raw Score={r['score']:.4f} | Normalized={normalized:.1f}/100] {r['text']}")
    else:
        print("No results found.")
