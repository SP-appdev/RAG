import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# SambaNova API setup
# ----------------------------
SAMBA_API_KEY = "95bbdcbb-46ab-451a-b341-e284e7ca9f4e"
LLM_MODEL = "Meta-llama-3" 

def generate_answer(prompt: str, llm_model: str = LLM_MODEL) -> str:
    """Generate answer using SambaNova LLM via OpenAI-compatible API"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=SAMBA_API_KEY, base_url="https://api.sambanova.ai/v1")

        if llm_model == "Llama-4-Maverick":
            model_name = "Llama-4-Maverick-17B-128E-Instruct"
        elif llm_model == "Meta-llama-3":
            model_name = "Meta-Llama-3.3-70B-Instruct"
        else:
            raise ValueError(f"Unsupported model: {llm_model}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating answer: {e}"

# ----------------------------
# Load JSON and chunk text
# ----------------------------
with open("data/clinic_info.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
for section, content in data.items():
    for key, value in content.items():
        texts.append(f"{section} - {key}: {value}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.create_documents(texts)

# ----------------------------
# Create embeddings and vector store
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# ----------------------------
# RAG query function
# ----------------------------
def answer(question: str) -> str:
    hits = vectorstore.similarity_search(question, k=3)
    context = "\n".join([h.page_content for h in hits])

    if not context.strip():
        return "I do not have this information."

    # Build prompt for LLM
    prompt = f"""
You are a helpful clinical assistant.
Use ONLY the context below to answer the question.
If the answer cannot be found, respond: "I do not have this information."

Context:
{context}

Question: {question}
Answer:
"""
    return generate_answer(prompt, llm_model=LLM_MODEL)

# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    print("Clinical RAG Assistant using SambaNova LLM. Type 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() == "exit":
            break
        print("Answer:", answer(q))
