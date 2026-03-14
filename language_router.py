"""
Language detection and routing for multilingual TTS system.
"""
import fasttext
import os
import warnings
warnings.filterwarnings('ignore')

class LanguageRouter:
    """Detects language and routes to appropriate TTS model."""
    
    # Language code mapping - use absolute path from workspace root
    # Get the workspace root (where models/ folder is located)
    WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    
    SUPPORTED_LANGUAGES = {
        'kn': os.path.join(WORKSPACE_ROOT, 'models', 'kn'),
        'hi': os.path.join(WORKSPACE_ROOT, 'models', 'hi'),
        'en': os.path.join(WORKSPACE_ROOT, 'models', 'en'),
        'ta': os.path.join(WORKSPACE_ROOT, 'models', 'ta'),
        'te': os.path.join(WORKSPACE_ROOT, 'models', 'te')
    }
    
    def __init__(self):
        """Initialize language detector."""
        self.model = None
        self._load_fasttext_model()
    
    def _load_fasttext_model(self):
        """Load fastText language identification model."""
        try:
            # Download model if not exists
            model_path = 'lid.176.bin'
            if not os.path.exists(model_path):
                print("Downloading fastText language detection model...")
                import urllib.request
                url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
                urllib.request.urlretrieve(url, model_path)
            
            self.model = fasttext.load_model(model_path)
            print("Language detection model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load fastText model: {e}")
            self.model = None
    
    def detect_language(self, text):
        """
        Detect language from input text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code (kn, hi, en, ta, te)
        """
        if not self.model:
            # Fallback: simple character-based detection
            return self._fallback_detection(text)
        
        try:
            # fastText prediction
            predictions = self.model.predict(text.replace('\n', ' '))
            lang_code = predictions[0][0].replace('__label__', '')
            
            # Map to supported languages
            if lang_code in self.SUPPORTED_LANGUAGES:
                return lang_code
            
            # Fallback for unsupported languages
            return self._fallback_detection(text)
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
        # Check Unicode ranges for Indic scripts
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
        
        # Default to English
        return 'en'
    
    def get_model_path(self, language_code):
        """
        Get model directory path for language.
        
        Args:
            language_code (str): Language code
            
        Returns:
            str: Path to model directory
        """
        return self.SUPPORTED_LANGUAGES.get(language_code, 'models/en')
