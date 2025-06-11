import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Step 1: Load Resume Text
def load_resume(filepath="resume.txt", chunk_size=512):
    with open(filepath, "r", encoding="utf-8") as f:
        resume = f.read()
    return [resume[i:i+chunk_size] for i in range(0, len(resume), chunk_size)]


# Step 2: Generate Embeddings
def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings


# Step 3: Build FAISS Index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Step 4: Retrieve Relevant Chunks
def retrieve_context(question, embedder, index, chunks, k=3):
    q_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(q_embedding), k)
    return [chunks[i] for i in indices[0]]


# Step 5: Generate Answer Using LLM
def generate_answer(question, context, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    prompt = (
        "Answer the question based on the provided context.\n\n"
        "Context:\n" + "\n".join(context) + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer[len(prompt):].strip()  # Only return generated part


# Main Function
def main():
    print("Loading resume...")
    chunks = load_resume()
    print(f"Loaded {len(chunks)} chunks.")

    print("Generating embeddings...")
    embedder, embeddings = generate_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        print("Retrieving context...")
        context = retrieve_context(question, embedder, index, chunks)

        print("Generating answer...")
        answer = generate_answer(question, context)

        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()