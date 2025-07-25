# 🧠 GenAI Text Summarizer

This is a simple, no-cost Generative AI app that summarizes long-form text using a pre-trained transformer model from Hugging Face.  
Built with 🤗 `facebook/bart-large-cnn` and deployed via `Gradio`, it allows users to enter long paragraphs and get concise summaries instantly.

---

## 🚀 Features

- 💡 Uses `facebook/bart-large-cnn` model for high-quality summarization
- ⚡ Real-time web interface using Gradio
- 📦 Runs locally or on Google Colab — no API key or paid model required
- ✅ Clean and minimal, perfect for GenAI demo and resume projects

---

## 📷 Demo Screenshot

![App Screenshot](demo.png)

---

## 🛠️ Installation

### ✅ Option 1: Run Locally

```bash
git clone https://github.com/your-username/text-summarizer-genai.git
cd text-summarizer-genai
pip install -r requirements.txt
python summarizer_app.py
