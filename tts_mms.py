"""
Fast and accurate TTS using Facebook's MMS (Massively Multilingual Speech).
Lightweight VITS-based model optimized for speed.
"""
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from scipy import signal

class MMSTTS:
    """
    Fast multilingual TTS using Facebook's MMS.
    
    Features:
    - Fast generation (2-4 seconds on CPU)
    - Good accuracy (~75-80%)
    - Lightweight models (~150MB each)
    - VITS-based (high quality)
    """
    
    def __init__(self):
        """Initialize MMS TTS."""
        print("Loading MMS TTS (fast, lightweight)...")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Language to model mapping
        self.language_models = {
            'kn': 'facebook/mms-tts-kan',  # Kannada
            'hi': 'facebook/mms-tts-hin',  # Hindi
            'en': 'facebook/mms-tts-eng',  # English
            'ta': 'facebook/mms-tts-tam',  # Tamil
            'te': 'facebook/mms-tts-tel',  # Telugu
            'ml': 'facebook/mms-tts-mal',  # Malayalam
            'mr': 'facebook/mms-tts-mar',  # Marathi
            'gu': 'facebook/mms-tts-guj',  # Gujarati
            'bn': 'facebook/mms-tts-ben',  # Bengali
        }
        
        # Cache for loaded models
        self.models = {}
        self.tokenizers = {}
        
        print("✓ MMS TTS initialized")
        print("  Speed: Fast (2-4s per sentence)")
        print("  Accuracy: Good (~75-80%)")
        print("  Size: Lightweight (~150MB per language)")
    
    def load_model(self, language_code):
        """
        Load model for specific language.
        
        Args:
            language_code (str): Language code (kn, hi, en, etc.)
            
        Returns:
            tuple: (model, tokenizer)
        """
        # Return cached if available
        if language_code in self.models:
            return self.models[language_code], self.tokenizers[language_code]
        
        # Get model name
        if language_code not in self.language_models:
            print(f"⚠ Language {language_code} not supported, using English")
            language_code = 'en'
        
        model_name = self.language_models[language_code]
        
        print(f"Loading {language_code} model: {model_name}...")
        
        try:
            # Load model and tokenizer
            model = VitsModel.from_pretrained(model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Cache them
            self.models[language_code] = model
            self.tokenizers[language_code] = tokenizer
            
            print(f"✓ {language_code} model loaded")
            return model, tokenizer
            
        except Exception as e:
            print(f"✗ Failed to load {language_code} model: {e}")
            raise
    
    def generate_speech(self, text, language_code='kn', speaker='female'):
        """
        Generate speech from text.
        
        Args:
            text (str): Text to synthesize
            language_code (str): Language code
            speaker (str): Ignored (MMS has single voice per language)
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        # Load model for language
        model, tokenizer = self.load_model(language_code)
        
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Convert to numpy
        audio = output.squeeze().cpu().numpy()
        
        return audio
    
    def save_audio(self, wav, output_path="output.wav"):
        """
        Save audio with quality processing.
        
        Args:
            wav (numpy.ndarray): Audio waveform
            output_path (str): Output file path
        """
        # MMS uses 16kHz sample rate
        sample_rate = 16000
        
        # Convert to numpy array
        wav = np.array(wav, dtype=np.float32)
        
        # 1. Remove DC offset
        wav = wav - np.mean(wav)
        
        # 2. High-pass filter (80Hz) - remove rumble
        nyquist = sample_rate / 2
        cutoff = 80 / nyquist
        b, a = signal.butter(4, cutoff, btype='high')
        wav = signal.filtfilt(b, a, wav)
        
        # 3. Low-pass filter (7500Hz) - remove noise (adjusted for 16kHz)
        cutoff_high = 7500 / nyquist
        b_high, a_high = signal.butter(4, cutoff_high, btype='low')
        wav = signal.filtfilt(b_high, a_high, wav)
        
        # 4. Normalize with soft clipping
        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val * 0.85
            wav = np.tanh(wav * 1.2) * 0.95
        
        # 5. Gentle silence removal
        threshold = 0.01
        non_silent = np.abs(wav) > threshold
        if non_silent.any():
            padding_samples = int(sample_rate * 0.05)  # 50ms
            start_idx = max(0, np.argmax(non_silent) - padding_samples)
            end_idx = min(len(wav), len(wav) - np.argmax(non_silent[::-1]) + padding_samples)
            wav = wav[start_idx:end_idx]
        
        # 6. Fade in/out
        fade_samples = int(sample_rate * 0.01)  # 10ms
        if len(wav) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            wav[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples)
            wav[-fade_samples:] *= fade_out
        
        # Save with high quality
        sf.write(output_path, wav, sample_rate, subtype='PCM_16')
        print(f"Audio saved to: {output_path}")


# Test function
if __name__ == "__main__":
    tts = MMSTTS()
    
    # Test Kannada
    print("\n--- Testing Kannada ---")
    text_kn = "ನಮಸ್ಕಾರ, ನಾನು ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಸಹಾಯಕ"
    wav = tts.generate_speech(text_kn, 'kn')
    tts.save_audio(wav, "test_mms_kannada.wav")
    print("✓ Kannada test complete")
    
    # Test Hindi
    print("\n--- Testing Hindi ---")
    text_hi = "नमस्ते, मैं आपका सहायक हूं"
    wav = tts.generate_speech(text_hi, 'hi')
    tts.save_audio(wav, "test_mms_hindi.wav")
    print("✓ Hindi test complete")
    
    # Test English
    print("\n--- Testing English ---")
    text_en = "Hello, I am your AI assistant"
    wav = tts.generate_speech(text_en, 'en')
    tts.save_audio(wav, "test_mms_english.wav")
    print("✓ English test complete")
