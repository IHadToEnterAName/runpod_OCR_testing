
# Corporate Document Assistant Using Runpod GPU Server (RAG + Vision)

This repository contains a professional AI assistant designed for corporate document analysis and information retrieval. The system utilizes Retrieval-Augmented Generation (RAG) and Vision-Language Models to process text, structured documents, and visual data from PDF, DOCX, and image files. This was tested using a **RTX 5090 GPU**.

## Overview

The application enables users to upload corporate documents and query them using natural language. It performs the following technical operations:

* **Semantic Extraction:** Uses PyMuPDF and python-docx for high-fidelity text extraction.
* **Visual Analysis:** Employs Llama 3.2 Vision to perform OCR and describe visual elements such as charts and diagrams.
* **Efficient Indexing:** Implements recursive character splitting and batch embedding generation via Ollama.
* **Fast Retrieval:** Uses vectorized NumPy operations to calculate cosine similarity for relevant context retrieval.

---

## Technical Stack

* **Language Model:** Mistral-7B
* **Vision Model:** Llama-3.2-Vision
* **Embedding Model:** Nomic-Embed-Text
* **Orchestration:** Chainlit
* **Processing:** NumPy, PyMuPDF, LangChain Text Splitters

---

## Installation and Setup

### 1. Environment Configuration

It is highly recommended to use a Python virtual environment to isolate project dependencies.

**Create and Activate Virtual Environment:**

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate

```

On Windows:

```bash
python -m venv runpod_test
venv runpod_test/bin/activate

```

### 2. Dependency Installation

Once the virtual environment is activated, install the required libraries:

```bash
pip install -r requirements.txt

```

### 3. Model Configuration

Ensure you have [Ollama](https://ollama.com/) installed and running. Pull the necessary models before starting the application:

```bash
ollama pull mistral
ollama pull llama3.2-vision
ollama pull nomic-embed-text

```

---

## Deployment Instructions

### Local Execution

To run the application locally, execute the following command:

```bash
chainlit run chat.py

```

### RunPod Deployment

When deploying on a RunPod GPU instance, you must configure the host and port to allow external access through the proxy.

1. **Configure Base URL:**
Update the `base_url` variable in `chat.py` with your unique RunPod Proxy URL (e.g., `https://[POD_ID]-11434.proxy.runpod.net`).
2. **Execute App:**
```bash
chainlit run chat.py

```


3. **Access UI:**
Access the interface via the RunPod proxy link provided using the local host link.

---

## Usage Guide

1. **Initialization:** Upon starting the application, you will be prompted to upload your source files.
2. **Processing:** The system will automatically chunk the text and generate embeddings. If images are detected, the vision model will generate descriptions to be indexed.
3. **Interaction:** Enter your queries in the chat interface. The system will retrieve the most relevant document segments to provide a context-aware response.

---

## Project Structure

```text
├── chat.py             # Main application logic
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules
└── .chainlit/          # Chainlit configuration files

```
