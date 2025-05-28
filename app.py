import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# from transformers import is_torch_xla_available
def load_resume(filepath,chunk_size=512):
    with open (filepath , 'r', encoding='utf-8') as file:
        resume=file.read()
        chunks=[resume[i:i+chunk_size] for i in range(0, len(resume), chunk_size)]
        return chunks
def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model,np.array(embeddings)

def create_faiss_index(embeddings):
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def hugging_face(model_name='mistralai/Mistral-7B-Instruct-v0.1'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return tokenizer, model

def generate_response(question, chunks, embedder, index, tokenizer, model):
    q = embedder.encode([question])
    _, top_index = index.search(np.array(q), k=3)
    context = '\n'.join([chunks[i] for i in top_index[0]])
    prompt = f"Answer the question based on the context provided:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    print("Prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    print("Raw output:", outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def main():
    chunks = load_resume('resume.txt')
    print("Chunks loaded:", len(chunks))
    embedder, vectors = generate_embeddings(chunks)
    index = create_faiss_index(vectors)
    tokenizer, model = hugging_face()
    question = "What are frameworks and libraries are you familiar with?"
    answer = generate_response(question, chunks, embedder, index, tokenizer, model)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
