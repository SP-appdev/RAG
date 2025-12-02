# simple_rag.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# SambaNova LLM API
# ----------------------------
SAMBA_API_KEY = "95bbdcbb-46ab-451a-b341-e284e7ca9f4e"
LLM_MODEL = "Llama-4-Maverick"

def generate_answer(prompt: str) -> str:
    """Generate answer using SambaNova LLM"""
    from openai import OpenAI

    client = OpenAI(api_key=SAMBA_API_KEY, base_url="https://api.sambanova.ai/v1")

    if LLM_MODEL == "Llama-4-Maverick":
        model_name = "Llama-4-Maverick-17B-128E-Instruct"
    elif LLM_MODEL == "Meta-llama-3":
        model_name = "Meta-Llama-3.3-70B-Instruct"
    else:
        raise ValueError("Unsupported model")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
    )
    return response.choices[0].message.content.strip()

# ----------------------------
# Load JSON data
# ----------------------------
with open("data/clinic_info.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to list of strings
texts = []
for section, content in data.items():
    for key, value in content.items():
        texts.append(f"{section} - {key}: {value}")

# ----------------------------
# Chunk text manually (200 chars, 50 overlap)
# ----------------------------
chunk_size = 200
overlap = 50
chunks = []

for text in texts:
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

# ----------------------------
# Embed chunks using MiniLM
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# ----------------------------
# Create FAISS index
# ----------------------------
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# ----------------------------
# RAG function
# ----------------------------
def answer(question: str, top_k=3) -> str:
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, idx = index.search(q_vec, top_k)
    context = "\n".join([chunks[i] for i in idx[0]])

    if not context.strip():
        return "I do not have this information."

    prompt = f"""
You are a helpful clinical assistant.
Use ONLY the context below to answer the question.
If the answer cannot be found, respond: "I do not have this information."

Context:
{context}

Question: {question}
Answer:
"""
    return generate_answer(prompt)

# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    print("Simple Clinical RAG Assistant. Type 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() == "exit":
            break
        print("Answer:", answer(q))
