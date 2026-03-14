"""
Streamlit frontend for Multilingual AI Assistant.
"""
import streamlit as st
import sys
import os
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speech.whisper_stt import WhisperSTT
from language.detect_language import LanguageDetector
from nlp.indicbert_intent import IntentClassifier
from llm.sarvam_client import SarvamClient
from translation.indictrans_translate import IndicTranslator
from tts.multilingual_tts import MultilingualTTS
from audio.record_audio import AudioRecorder
from audio.play_audio import AudioPlayer

# Page configuration
st.set_page_config(
    page_title="Multilingual AI Assistant",
    page_icon="🎤",
    layout="centered"
)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models (cached)
@st.cache_resource
def load_models():
    """Load all models once."""
    print("Loading models...")
    
    # Sarvam AI API key — loaded from .env file or SARVAM_API_KEY env variable
    sarvam_api_key = None  # SarvamClient will auto-load from .env

    models = {
        'stt': WhisperSTT(model_size="small"),
        'language_detector': LanguageDetector(),
        'intent_classifier': IntentClassifier(),
        'llm': SarvamClient(model="sarvam-30b"),  # key loaded from .env
        'translator': IndicTranslator(),
        'tts': MultilingualTTS(),
        'recorder': AudioRecorder(),
        'player': AudioPlayer()
    }
    print("All models loaded!")
    return models

# Title
st.title("🎤 Multilingual AI Assistant")
st.caption("Powered by Sarvam AI • Native Indian Language Support")
st.markdown("---")

# Load models
with st.spinner("Loading models... (This may take a minute on first run)"):
    models = load_models()
    st.session_state.models_loaded = True

# Language and voice selection
st.markdown("### Language & Voice Settings")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**Supported Languages:**")
    st.markdown("🇮🇳 Kannada • Hindi • Tamil • Telugu • 🇬🇧 English")
with col2:
    speaker_voice = st.selectbox(
        "Voice",
        ["female", "male", "female_slow", "male_slow"],
        format_func=lambda x: {
            "female": "👩 Female",
            "male": "👨 Male",
            "female_slow": "👩 Female (Clear/Slow)",
            "male_slow": "👨 Male (Clear/Slow)"
        }[x]
    )
with col3:
    # Voice preview button
    if st.button("🔊 Preview Voice"):
        with st.spinner("Generating preview..."):
            try:
                # Sample text in different languages
                preview_texts = {
                    'kn': 'ನಮಸ್ಕಾರ, ನಾನು ನಿಮ್ಮ ಸಹಾಯಕ',
                    'hi': 'नमस्ते, मैं आपका सहायक हूं',
                    'en': 'Hello, I am your assistant',
                    'ta': 'வணக்கம், நான் உங்கள் உதவியாளர்',
                    'te': 'నమస్కారం, నేను మీ సహాయకుడిని'
                }
                
                # Use first language for preview (or detect from last conversation)
                preview_lang = 'en'
                if st.session_state.conversation:
                    last_lang = st.session_state.conversation[-1].get('language', 'en')
                    preview_lang = last_lang
                
                preview_text = preview_texts.get(preview_lang, preview_texts['en'])
                
                # Generate preview audio
                preview_path = models['tts'].synthesize(
                    preview_text,
                    preview_lang,
                    speaker=speaker_voice,
                    output_path="preview.wav"
                )
                
                if preview_path and os.path.exists(preview_path):
                    st.audio(preview_path, format='audio/wav')
                    st.success(f"Preview: {speaker_voice.title()} voice in {preview_lang.upper()}")
            except Exception as e:
                st.error(f"Preview error: {e}")

st.markdown("---")

# Input section
st.markdown("### Input")

# Text input
user_input = st.text_input("Type your message:", placeholder="Type here or use voice input...")

# Voice input button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎤 Record Voice (5 seconds)", use_container_width=True):
        with st.spinner("Recording..."):
            try:
                # Record audio
                audio_data = models['recorder'].record(duration=5)
                audio_file = models['recorder'].save(audio_data, "temp_recording.wav")
                
                # Transcribe
                with st.spinner("Transcribing..."):
                    result = models['stt'].transcribe(audio_file)
                    user_input = result['text']
                    st.success(f"Transcribed: {user_input}")
            except Exception as e:
                st.error(f"Recording error: {e}")

# Process button
if st.button("💬 Send", use_container_width=True):
    if user_input:
        # Prevent duplicate processing
        if 'last_processed' not in st.session_state or st.session_state.last_processed != user_input:
            st.session_state.last_processed = user_input
            
            start_time = time.time()
            
            with st.spinner("Processing..."):
                try:
                    # Step 1: Detect language
                    detected_lang = models['language_detector'].detect(user_input)
                    st.info(f"Detected language: {detected_lang}")
                    
                    # Step 2: Classify intent
                    intent = models['intent_classifier'].classify(user_input)
                    st.info(f"Intent: {intent}")
                    
                    # Step 4: Generate response with LLM
                    with st.spinner("Generating response (first query may take 30-60 seconds)..."):
                        # Sarvam AI responds in the same language as the query
                        llm_response = models['llm'].generate(user_input, intent, detected_lang)
                    
                    # Step 5: Use detected input language for TTS
                    # Sarvam AI responds in the same language as the query
                    response_lang = detected_lang
                    st.info(f"Response language: {response_lang}")
                    final_response = llm_response
                
                    # Step 6: Generate speech in detected response language
                    with st.spinner("Generating speech..."):
                        audio_path = models['tts'].synthesize(
                            final_response,
                            response_lang,  # Use detected response language
                            speaker=speaker_voice,
                            output_path="response.wav"
                        )
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    
                    # Add to conversation
                    st.session_state.conversation.append({
                        'user': user_input,
                        'assistant': final_response,
                        'language': detected_lang,
                        'intent': intent,
                        'time': total_time,
                        'audio': audio_path
                    })
                    
                    # Display response
                    st.success(f"✅ Response generated in {total_time:.2f} seconds")
                    
                    # Play audio - Use ONLY Streamlit's audio player to avoid echo
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path, format='audio/wav', autoplay=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

st.markdown("---")

# Conversation history
st.markdown("### Conversation History")

if st.session_state.conversation:
    for i, msg in enumerate(reversed(st.session_state.conversation)):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**{msg['language'].upper()}**")
                st.caption(f"{msg['intent']}")
            with col2:
                st.markdown(f"**You:** {msg['user']}")
                st.markdown(f"**Assistant:** {msg['assistant']}")
                if msg.get('audio') and os.path.exists(msg['audio']):
                    st.audio(msg['audio'], format='audio/wav')
                st.caption(f"⏱️ {msg['time']:.2f}s")
            st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.conversation = []
        st.rerun()
else:
    st.info("No conversation yet. Start by typing or recording a message!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Powered by Sarvam AI • Whisper • IndicTTS • Running with cloud AI</small>
</div>
""", unsafe_allow_html=True)
