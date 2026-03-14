"""
Speech-to-text using Whisper model.
"""
import whisper
import numpy as np

class WhisperSTT:
    """Speech recognition using Whisper."""
    
    def __init__(self, model_size="small"):
        """
        Initialize Whisper model.
        
        Args:
            model_size (str): Model size (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded")
    
    def transcribe(self, audio_path_or_data):
        """
        Transcribe audio to text.
        
        Args:
            audio_path_or_data: Path to audio file or numpy array
            
        Returns:
            dict: Transcription result with text and detected language
        """
        print("Transcribing audio...")
        
        # Transcribe with language detection
        result = self.model.transcribe(
            audio_path_or_data,
            fp16=False,  # Use FP32 for CPU compatibility
            language=None  # Auto-detect language
        )
        
        text = result['text'].strip()
        detected_lang = result.get('language', 'unknown')
        
        print(f"Transcription: {text}")
        print(f"Detected language: {detected_lang}")
        
        return {
            'text': text,
            'language': detected_lang
        }
