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
        import soundfile as sf
        sample_rate = 16000
        wav = np.array(wav, dtype=np.float32)
        wav = wav - np.mean(wav)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.90
        fade = int(sample_rate * 0.005)
        if len(wav) > fade * 2:
            wav[:fade]  *= np.linspace(0, 1, fade)
            wav[-fade:] *= np.linspace(1, 0, fade)
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
