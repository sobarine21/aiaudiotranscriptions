import streamlit as st
import requests
from textblob import TextBlob
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from collections import Counter
from pydub import AudioSegment
from io import BytesIO
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Retrieve Hugging Face API token from Streamlit secrets
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to send the audio file to the API
def transcribe_audio(file):
    try:
        # Read the file as binary
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            return response.json()  # Return transcription
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Function for keyword extraction using NLTK
def extract_keywords(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]  # Remove non-alphanumeric characters
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stopwords]
    freq_dist = FreqDist(words)
    most_common_words = freq_dist.most_common(10)
    return most_common_words

# Function to simulate speaker detection (based on pauses in speech, for now, this is a placeholder)
def detect_speakers(audio_file):
    # For simplicity, we'll simulate speaker detection by splitting audio by silence
    audio = AudioSegment.from_file(audio_file)
    silence_thresh = audio.dBFS - 14  # Adjust for sensitivity
    segments = audio.split_to_mono()  # This will split into individual channels if multiple speakers
    speakers = []
    for i, segment in enumerate(segments):
        # Placeholder: This is where you'd implement actual speaker diarization.
        speakers.append(f"Speaker {i+1}: {len(segment)} ms")
    return speakers

# Streamlit UI
st.title("🎙️ Audio Transcription & Analysis Web App")
st.write("Upload an audio file, and this app will transcribe it using OpenAI Whisper via Hugging Face API.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    st.info("Transcribing audio... Please wait.")
    
    # Transcribe the uploaded audio file
    result = transcribe_audio(uploaded_file)
    
    # Display the result
    if "text" in result:
        st.success("Transcription Complete:")
        transcription_text = result["text"]
        st.write(transcription_text)
        
        # Perform Sentiment Analysis
        sentiment = analyze_sentiment(transcription_text)
        st.subheader("Sentiment Analysis")
        st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

        # Perform Keyword Extraction
        keywords = extract_keywords(transcription_text)
        st.subheader("Keyword Extraction")
        st.write(keywords)

        # Speaker Detection (placeholder for actual implementation)
        speakers = detect_speakers(uploaded_file)
        st.subheader("Speaker Detection (Placeholder)")
        st.write(speakers)
        
        # Add download button for the transcription text
        st.download_button(
            label="Download Transcription",
            data=transcription_text,
            file_name="transcription.txt",
            mime="text/plain"
        )
        
        # Add download button for analysis results
        analysis_results = f"""
        Sentiment Analysis:
        Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}
        
        Keyword Extraction:
        {keywords}
        """
        st.download_button(
            label="Download Analysis Results",
            data=analysis_results,
            file_name="analysis_results.txt",
            mime="text/plain"
        )
        
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
