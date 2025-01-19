import streamlit as st
import os
import wave
import pyaudio
import whisper
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize session state variables
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = whisper.load_model("base", device="cpu")  # Whisper "base" model for accuracy

if "sentiment_model" not in st.session_state:
    st.session_state.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased")

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "llm_model" not in st.session_state:
    st.session_state.llm_model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Llama3-8b-8192"
    )

if "call_summaries" not in st.session_state:
    st.session_state.call_summaries = []  # Store summaries of previous calls

# Function to record audio
def record_audio(filename="output.wav", duration=5, rate=16000, chunk=1024):
    """
    Record audio using PyAudio and save it as a WAV file.
    """
    try:
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
    except Exception as e:
        st.error(f"Error while recording audio: {e}")

# Function to transcribe audio
def transcribe_audio(filename="output.wav"):
    """
    Transcribe audio using Whisper and return the transcription text.
    """
    try:
        model = st.session_state.whisper_model
        result = model.transcribe(filename)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

# Perform sentiment analysis
def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text.
    """
    try:
        sentiment_result = st.session_state.sentiment_model(text[:500])  # Limit to 500 characters for speed
        return sentiment_result[0]  # Return the first result
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {}

# Function to process PDF and create FAISS vector embeddings
def vector_embedding(pdf_path):
    """
    Embed PDF documents into FAISS for RAG-based retrieval.
    """
    try:
        if not os.path.exists(pdf_path):
            st.error(f"File not found: {pdf_path}")
            return

        if st.session_state.vectors is None:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split the Document objects into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            final_documents = text_splitter.split_documents(docs)

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
            st.session_state.vectors.save_local("faiss_index")  # Save for reuse

        st.success("Document embedding completed successfully!")
    except Exception as e:
        st.error(f"Error during embedding: {e}")

# Load FAISS index
def load_faiss_index():
    """
    Load the FAISS index from disk if not already in session state.
    """
    if st.session_state.vectors is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.vectors = FAISS.load_local("faiss_index", embeddings)

# Generate response using RAG
def generate_response(input_text):
    """
    Generate a response using RAG (Retrieval-Augmented Generation).
    """
    try:
        load_faiss_index()  # Ensure FAISS index is loaded
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents
        llm = st.session_state.llm_model

        # Combine retrieval and LLM into a chain
        document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("{context}{input}"))
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': input_text})
        return response.get("answer", "No response generated.")
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "No response generated."

# Display the dashboard
def display_dashboard():
    """
    Display the dashboard with summaries of all previously recorded calls.
    """
    st.subheader("Call Summary Dashboard")

    # Show previous call summaries
    st.markdown("### Previous Call Summaries")
    if st.session_state.call_summaries:
        for idx, summary in enumerate(st.session_state.call_summaries, start=1):
            st.markdown(f"#### Call {idx}")
            st.markdown(f"- **Transcription**: {summary['transcription']}")
            st.markdown(f"- **Sentiment**: {summary['sentiment']['label']}")
            st.markdown(f"- **Confidence**: {summary['sentiment']['score']:.2f}")
            st.markdown(f"- **Generated Response**: {summary['response']}")
            st.write("---")
    else:
        st.write("No call summaries available.")

# Streamlit app with tabs for recording and dashboard
st.title("Call Recording and RAG Summary Dashboard")

# Create tabs for recording and dashboard
tab1, tab2 = st.tabs(["📞 Record a Call", "📊 Dashboard"])

# Tab 1: Record a call
with tab1:
    st.subheader("Record and Analyze a New Call")
    duration = st.slider("Select Recording Duration (seconds)", min_value=5, max_value=60, value=10, step=5)

    if st.button("Record and Analyze"):
        # Record audio
        audio_path = "output.wav"
        record_audio(audio_path, duration=duration)  # Record audio for user-selected duration

        # Transcribe audio
        transcription = transcribe_audio(audio_path)
        st.write("Transcribed Text:", transcription)

        # Perform Sentiment Analysis
        sentiment_result = analyze_sentiment(transcription)
        st.write("Sentiment Analysis Result:", sentiment_result)

        # Generate response using RAG
        response = generate_response(transcription)
        st.write("Generated Response:", response)

        # Save the summary
        summary = {
            "transcription": transcription,
            "sentiment": sentiment_result,
            "response": response
        }
        st.session_state.call_summaries.append(summary)
        st.success("Call summary saved!")

# Tab 2: Dashboard
with tab2:
    display_dashboard()
