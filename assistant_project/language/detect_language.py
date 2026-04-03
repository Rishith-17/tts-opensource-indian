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
        """Detect language — Unicode script check takes priority over fastText."""
        # Unicode check first — 100% accurate for Indic scripts
        script_lang = self._fallback_detection(text)
        if script_lang != 'en':
            return script_lang   # Kannada/Hindi/Tamil/Telugu detected by script

        # English text — use fastText for confidence
        if not self.model:
            return 'en'
        try:
            predictions = self.model.predict(text.replace('\n', ' '))
            lang_code = predictions[0][0].replace('__label__', '')
            return self.LANG_MAP.get(lang_code, 'en')
        except Exception:
            return 'en'
    
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
