"""
Language detection using fastText.
"""
import fasttext
import os
import warnings
warnings.filterwarnings('ignore')

class LanguageDetector:
    """Detects language from text."""
    
    # Language code mapping
    LANG_MAP = {
        'kn': 'kn',  # Kannada
        'hi': 'hi',  # Hindi
        'en': 'en',  # English
        'ta': 'ta',  # Tamil
        'te': 'te',  # Telugu
        'kan': 'kn',
        'hin': 'hi',
        'eng': 'en',
        'tam': 'ta',
        'tel': 'te'
    }
    
    def __init__(self):
        """Initialize language detector."""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load fastText language identification model."""
        try:
            model_path = 'lid.176.bin'
            if not os.path.exists(model_path):
                print("Downloading fastText language detection model...")
                import urllib.request
                url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
                urllib.request.urlretrieve(url, model_path)
            
            self.model = fasttext.load_model(model_path)
            print("Language detection model loaded")
        except Exception as e:
            print(f"Warning: Could not load fastText model: {e}")
            self.model = None
    
    def detect(self, text):
        """
        Detect language from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code (kn, hi, en, ta, te)
        """
        if not self.model:
            return self._fallback_detection(text)
        
        try:
            predictions = self.model.predict(text.replace('\n', ' '))
            lang_code = predictions[0][0].replace('__label__', '')
            
            # Map to supported languages
            return self.LANG_MAP.get(lang_code, self._fallback_detection(text))
        except Exception as e:
            print(f"Detection error: {e}")
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text):
        """
        Simple character-based language detection fallback.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code
        """
        for char in text:
            code = ord(char)
            if 0x0C80 <= code <= 0x0CFF:  # Kannada
                return 'kn'
            elif 0x0900 <= code <= 0x097F:  # Devanagari (Hindi)
                return 'hi'
            elif 0x0B80 <= code <= 0x0BFF:  # Tamil
                return 'ta'
            elif 0x0C00 <= code <= 0x0C7F:  # Telugu
                return 'te'
        
        return 'en'  # Default to English
