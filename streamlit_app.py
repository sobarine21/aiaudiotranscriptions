import streamlit as st
import requests
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import langid
from collections import Counter
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Retrieve API tokens from Streamlit secrets
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to cycle through available Gemini models and corresponding API keys
def get_next_model_and_key():
    models_and_keys = [
        ('gemini-1.5-flash', os.getenv("API_KEY_GEMINI_1_5_FLASH")),
        ('gemini-2.0-flash', os.getenv("API_KEY_GEMINI_2_0_FLASH")),
        ('gemini-1.5-flash-8b', os.getenv("API_KEY_GEMINI_1_5_FLASH_8B")),
        ('gemini-2.0-flash-exp', os.getenv("API_KEY_GEMINI_2_0_FLASH_EXP")),
    ]
    for model, key in models_and_keys:
        if key:
            return model, key
    return None, None

# Function to send the audio file to the API
def transcribe_audio(file):
    try:
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

# Enhanced sentiment analysis with VADER
def analyze_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Function for keyword extraction using CountVectorizer
def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Function to calculate speech rate (words per minute)
def calculate_speech_rate(text, duration_seconds):
    words = text.split()
    num_words = len(words)
    return num_words / (duration_seconds / 60) if duration_seconds > 0 else 0

# Function to analyze call sentiment over time (simulated)
def analyze_sentiment_over_time(text):
    sentences = text.split('.')
    return [TextBlob(sentence).sentiment.polarity for sentence in sentences if sentence]

# Detect language of the text
def detect_language(text):
    return langid.classify(text)

# Function to calculate word frequency
def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(20)

# Function to generate a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Function to distill text by extracting important sentences
def distill_text(text, num_sentences=5):
    blob = TextBlob(text)
    sentences = blob.sentences
    scored_sentences = sorted(sentences, key=lambda s: s.sentiment.polarity, reverse=True)
    return ' '.join([str(sentence) for sentence in scored_sentences[:num_sentences]])

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
        
        # Distill text
        distilled_text = distill_text(transcription_text)
        st.subheader("Distilled Text")
        st.write(distilled_text)
        
        # Sentiment Analysis (TextBlob)
        sentiment = TextBlob(distilled_text).sentiment
        st.subheader("Sentiment Analysis (TextBlob)")
        st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

        # Sentiment Analysis (VADER)
        vader_sentiment = analyze_vader_sentiment(distilled_text)
        st.subheader("Sentiment Analysis (VADER)")
        st.write(f"Positive: {vader_sentiment['pos']}, Neutral: {vader_sentiment['neu']}, Negative: {vader_sentiment['neg']}")
        
        # Language Detection
        lang, confidence = detect_language(distilled_text)
        st.subheader("Language Detection")
        st.write(f"Detected Language: {lang}, Confidence: {confidence}")

        # Keyword Extraction
        keywords = extract_keywords(distilled_text)
        st.subheader("Keyword Extraction")
        st.write(keywords)

        # Speech Rate Calculation (using audio file duration)
        duration_seconds = len(uploaded_file.read()) / (44100 * 2)  # Estimate based on sample rate (44100 Hz)
        speech_rate = calculate_speech_rate(distilled_text, duration_seconds)
        st.subheader("Speech Rate")
        st.write(f"Speech Rate: {speech_rate} words per minute")

        # Sentiment Analysis Over Time
        sentiment_over_time = analyze_sentiment_over_time(distilled_text)
        st.subheader("Sentiment Analysis Over Time")
        st.line_chart(sentiment_over_time)

        # Word Frequency Analysis
        word_freq = word_frequency(distilled_text)
        st.subheader("Word Frequency Analysis")
        st.write(word_freq)

        # Word Cloud Visualization
        wordcloud = generate_word_cloud(distilled_text)
        st.subheader("Word Cloud")
        st.image(wordcloud.to_array())

        # Plot sentiment distribution
        st.subheader("Sentiment Distribution")
        plt.hist(sentiment_over_time, bins=20, color='blue', alpha=0.7)
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        st.pyplot(plt.gcf())
        
        # Add download button for the transcription text
        st.download_button(
            label="Download Transcription",
            data=transcription_text,
            file_name="transcription.txt",
            mime="text/plain"
        )
        
        # Add download button for analysis results
        analysis_results = f"""
        Sentiment Analysis (TextBlob):
        Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}
        
        Sentiment Analysis (VADER):
        Positive: {vader_sentiment['pos']}, Neutral: {vader_sentiment['neu']}, Negative: {vader_sentiment['neg']}
        
        Language Detection:
        Detected Language: {lang}, Confidence: {confidence}
        
        Keyword Extraction:
        {keywords}
        
        Speech Rate: {speech_rate} words per minute
        """
        st.download_button(
            label="Download Analysis Results",
            data=analysis_results,
            file_name="analysis_results.txt",
            mime="text/plain"
        )

        # Generative AI Analysis
        st.subheader("Generative AI Analysis")
        prompt = f"Analyze the following call recording transcription for professional call audit purposes in precise way and focus on highlighting the support agents KPI & metrics give numbers and scores for the performance on the metrics : {distilled_text}"
        
        # Let user decide if they want to use AI analysis
        if st.button("Run AI Analysis"):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                st.write("AI Analysis Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
