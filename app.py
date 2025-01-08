import pyaudio
import wave
import whisper
import pandas as pd
import sqlite3
import streamlit as st
from transformers import pipeline
from datetime import datetime

# --- Audio Parameters ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# --- Whisper Model Loading ---
model = whisper.load_model("base")

# --- PyAudio Initialization ---
audio = pyaudio.PyAudio()

# --- Load Sentiment Analysis Model ---
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# --- Database Setup ---
def setup_database():
    """Creates the necessary SQLite tables if they do not exist."""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sentiment TEXT,
            summary TEXT,
            customer_id TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            timestamp TEXT,
            chunk_text TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

setup_database()

def record_audio():
    """Records audio for the specified duration."""
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    return b''.join(frames)

def analyze_sentiment(audio_data):
    """Analyzes the sentiment of the recorded audio."""
    with wave.open("temp.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

    result = model.transcribe("temp.wav")
    text = result["text"]

    # --- Sentiment Analysis using BERT ---
    sentiment_result = sentiment_analyzer(text)
    sentiment = sentiment_result[0]["label"]

    return sentiment, text

def detect_sentiment_shift(previous_sentiment, current_sentiment):
    """Detects sentiment shifts."""
    return previous_sentiment != current_sentiment

def store_chunk(conversation_id, timestamp, chunk_text):
    """Stores individual chunks of a conversation in the database."""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversation_chunks (conversation_id, timestamp, chunk_text) VALUES (?, ?, ?)",
              (conversation_id, timestamp, chunk_text))
    conn.commit()
    conn.close()

def store_conversation_summary(timestamp, sentiment, summary, customer_id):
    """Stores the summary of a conversation in the database."""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (timestamp, sentiment, summary, customer_id) VALUES (?, ?, ?, ?)",
              (timestamp, sentiment, summary, customer_id))
    conn.commit()
    conversation_id = c.lastrowid
    conn.close()
    return conversation_id

def generate_conversation_summary(chunks):
    """Generates a summary of the entire conversation."""
    full_text = " ".join(chunks)
    summary_prompt = f"Summarize the following conversation: {full_text}"
    response = sentiment_analyzer(summary_prompt)
    return response[0]["label"], full_text  # Replace with appropriate summary handling

def main():
    st.title("Real-time Speech Analysis")

    # --- Streamlit UI Elements ---
    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording")

    # --- State Variables ---
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []

    if 'previous_sentiment' not in st.session_state:
        st.session_state.previous_sentiment = "neutral"

    if start_button:
        st.write("Recording...")

        audio_data = record_audio()

        current_sentiment, chunk_text = analyze_sentiment(audio_data)
        st.session_state.chunks.append(chunk_text)

        # Store chunk in database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = store_conversation_summary(timestamp, "neutral", "", "customer_001")

        store_chunk(st.session_state.conversation_id, timestamp, chunk_text)

        # Detect sentiment shift
        if detect_sentiment_shift(st.session_state.previous_sentiment, current_sentiment):
            st.success(f"Sentiment Shift Detected: {st.session_state.previous_sentiment} -> {current_sentiment}")

        st.session_state.previous_sentiment = current_sentiment

        st.write(f"Sentiment: {current_sentiment}")
        st.write(f"Chunk Text: {chunk_text}")

    if stop_button:
        st.write("Recording stopped.")

        # Generate and store conversation summary
        sentiment, summary = generate_conversation_summary(st.session_state.chunks)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store_conversation_summary(timestamp, sentiment, summary, "customer_001")

        st.write("Summary of the conversation:")
        st.write(summary)

if __name__ == "__main__":
    main()
