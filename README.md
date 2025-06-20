
# 📄 RAG Chatbot for Resume Analysis

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on the content of a resume (PDF file). This project uses Hugging Face models for embeddings and generation, along with FAISS for efficient similarity search.

## 🔍 Features

- Uploads and parses a resume in PDF format.
- Uses `sentence-transformers/all-mpnet-base-v2` for generating document embeddings.
- Leverages FAISS for fast vector search.
- Uses the `HuggingFaceH4/zephyr-7b-beta` model via Hugging Face Inference API for question answering.
- Answers predefined questions like CGPA, expertise, and projects.

---

## 🧰 Technologies Used

- [LangChain](https://www.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss) (by Meta/Facebook AI)
- [PyPDF](https://pypi.org/project/pypdf/)

---

## 📦 Requirements

Before running the application, make sure you have:

- Python 3.8+
- A Hugging Face account and API token

---

## 🛠️ Installation

```bash
pip install -U langchain langchain-huggingface sentence-transformers faiss-cpu transformers pypdf
```

> ⚠️ If using this in Google Colab or Jupyter Notebook, restart the runtime after installing dependencies.

---

## 🔐 Get Your Hugging Face API Token

1. Go to [Hugging Face](https://huggingface.co/) and sign up or log in.
2. Click on your profile icon in the top-right corner.
3. Select **Settings** → **Access Tokens**.
4. Copy your **Write** token (you’ll need it for inference).
5. Replace `"your_hugging_face_token"` in the script with your actual token.

---

## 📁 Usage Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Hari-Kec/Chatbot-for-Resume.git 
   cd Chatbot-for-Resume
   ```

2. Run the script in a Jupyter notebook or Google Colab environment.

3. Upload your resume when prompted.

4. The chatbot will automatically answer the following questions:

   - What is the current CGPA?
   - What is the candidate's area of expertise?
   - List any projects mentioned in the document.

---

## 💬 Example Output

```text
QUESTION: What is the current CGPA?
ANSWER: The candidate's current CGPA is 3.8 out of 4.0.

QUESTION: What is the candidate's area of expertise?
ANSWER: The candidate specializes in machine learning, natural language processing, and full-stack web development.

QUESTION: List any projects mentioned in the document.
ANSWER: The projects include Sentiment Analyzer using BERT, Real-time Chat Application, and Image Classification with CNN.
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✅ Contributing

Contributions are welcome! Please open an issue or submit a pull request if you'd like to help improve this project.

---

## 📬 Contact

For questions or suggestions, feel free to reach out to me at your.email@example.com or [LinkedIn Profile](https://linkedin.com/in/yourprofile).

---

### Happy coding! 👨‍💻📄🤖

--- 

Let me know if you want a version that works locally with a Streamlit or Gradio UI!
