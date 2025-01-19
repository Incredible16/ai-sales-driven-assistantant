import streamlit as st
import os
import time
import ollama  # Ensure Ollama is installed correctly
import pyaudio
import wave
import whisper
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Function to load the Ollama model
def load_ollama_model():
    """
    Interacts with the Ollama model via the Python SDK.
    """
    try:
        return ollama
    except Exception as e:
        st.error(f"Error loading Ollama model: {e}")
        return None

# Initialize Ollama model
llm = load_ollama_model()

# Function to record audio using PyAudio
def record_audio(filename="output.wav", duration=5, rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    st.write(f"Recording for {duration} seconds...")
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    st.write(f"Audio saved to {filename}")

# Transcribe audio using Whisper
def transcribe_audio(filename="output.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

# Function to embed PDF into vector store
def vector_embedding(pdf_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.loader = PyPDFLoader(pdf_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Path to the PDF file
pdf_path = r"C:\Python files\ragaisalesagent\data\laptopprices.pdf"

# Embedding button
if st.button("Documents Embedding"):
    vector_embedding(pdf_path)
    st.write("Vector Store DB is ready")

# Record, transcribe audio, and generate response using Ollama
if st.button("Record, Transcribe, and Query"):
    audio_path = "output.wav"
    record_audio(audio_path, duration=5)
    transcribed_text = transcribe_audio(audio_path)
    st.write("Transcribed Text:", transcribed_text)

    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.time()
        response = retrieval_chain.invoke({'input': transcribed_text})
        st.write(f"Response time: {time.time() - start} seconds")
        st.write(response['answer'])

        # Display document context in an expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Function to save the FAISS vector store to a file
def save_vector_store(vector_store, filename="vector_store.pkl"):
    try:
        with open(filename, "wb") as f:
            pickle.dump(vector_store, f)
        st.success(f"Vector store saved as {filename}")
    except Exception as e:
        st.error(f"Error saving vector store: {e}")

# Add a download button for the vector store
if "vectors" in st.session_state:
    with open("vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)
        st.write("Vector store loaded successfully!")

    # Save metadata for later use
    documents = [doc.page_content for doc in st.session_state.final_documents]
    metadata = [doc.metadata for doc in st.session_state.final_documents]
    with open("vector_metadata.json", "w") as json_file:
        json.dump({"documents": documents, "metadata": metadata}, json_file, indent=4)
    st.session_state.vectors.save_local("faiss_index")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    st.write("No documents embedded. Using general assistant:")
    general_prompt_input = prompt_general.format_messages(input=transcribed_text)
    general_response = llm(general_prompt_input)
    st.write(general_response.content if hasattr(general_response, "content") else general_response)
