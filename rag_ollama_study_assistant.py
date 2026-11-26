"""
AI Study Assistant using RAG + Ollama
=====================================
Google Colab setup:
    !pip install gradio pypdf chromadb sentence-transformers requests
    !curl -fsSL https://ollama.com/install.sh | sh
    !ollama pull llama3.2

Start Ollama in a separate cell/background process before running:
    import subprocess, time
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

Then run:
    !python rag_ollama_study_assistant.py

This app processes one text-based PDF at a time. Scanned PDFs require OCR.
"""

import requests
import gradio as gr
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llama3.2"
COLLECTION_NAME = "study_notes"

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.create_collection(name=COLLECTION_NAME)


def split_text(text, chunk_size=900, overlap=150):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def upload_pdf(pdf_file):
    """Extract PDF text, embed chunks, and save them in ChromaDB."""
    global collection

    if pdf_file is None:
        return "Please upload a PDF first."

    try:
        reader = PdfReader(pdf_file)
        chunks, metadata = [], []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk in split_text(page_text):
                chunks.append(chunk)
                metadata.append({"page": page_number})

        if not chunks:
            return (
                "No readable text found. This may be a scanned PDF; "
                "run OCR and upload a searchable PDF."
            )

        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = chroma_client.create_collection(name=COLLECTION_NAME)

        vectors = embedding_model.encode(chunks, show_progress_bar=False)
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors.tolist(),
            metadatas=metadata
        )

        return (
            "PDF processed successfully!\n"
            f"Pages: {len(reader.pages)}\n"
            f"Chunks indexed: {len(chunks)}\n"
            "Vector database: ChromaDB"
        )

    except Exception as error:
        return f"PDF processing error: {error}"


def ask_question(question):
    """Retrieve relevant PDF chunks and generate a grounded answer with Ollama."""
    if not question or not question.strip():
        return "Please enter a question."

    if collection.count() == 0:
        return "Please upload and process a PDF first."

    try:
        query_vector = embedding_model.encode([question])[0]
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(4, collection.count())
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        if not docs:
            return "No relevant information was found in the PDF."

        context = "\n\n".join(
            f"Page {meta.get('page', '?')}: {doc}"
            for doc, meta in zip(docs, metas)
        )

        prompt = f"""
You are an AI Study Assistant. Answer using ONLY the retrieved context.
If the answer is not present, say:
"The information was not found in the uploaded document."
Do not invent facts. Explain in simple student-friendly language.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=240
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "Cannot connect to Ollama. Start the Ollama server and make sure "
            "the llama3.2 model has been downloaded."
        )
    except Exception as error:
        return f"Error: {error}"


with gr.Blocks(title="AI Study Assistant - RAG + Ollama") as demo:
    gr.Markdown(
        """
        # AI Study Assistant
        ### RAG + Ollama + Llama 3.2

        Upload a PDF, process it, and ask questions answered from your notes.
        """
    )

    pdf_input = gr.File(
        label="Upload PDF notes",
        file_types=[".pdf"],
        type="filepath"
    )
    process_button = gr.Button("Process PDF", variant="primary")
    status = gr.Textbox(label="Processing status", lines=4)
    process_button.click(upload_pdf, inputs=pdf_input, outputs=status)

    gr.Markdown("## Ask questions about your PDF")
    question = gr.Textbox(
        label="Your question",
        placeholder="Example: Explain inheritance in Java."
    )
    ask_button = gr.Button("Ask AI", variant="primary")
    answer = gr.Markdown()
    ask_button.click(ask_question, inputs=question, outputs=answer)

    gr.Markdown(
        "**Stack:** Python | RAG | Ollama | Llama 3.2 | "
        "ChromaDB | Sentence Transformers | PyPDF | Gradio"
    )


if __name__ == "__main__":
    print("Launching AI Study Assistant...")
    # share=True creates a temporary public link. Do not upload private PDFs.
    demo.launch(share=True)
