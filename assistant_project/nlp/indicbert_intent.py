"""
Intent detection using rule-based classification.
For production, you can replace with IndicBERT v2 fine-tuned model.
"""
import re

class IntentClassifier:
    """Classifies user intent from text."""
    
    # Intent patterns
    INTENT_PATTERNS = {
        'greeting': [
            r'\b(hello|hi|hey|namaste|namaskar|vanakkam)\b',
            r'\b(ಹಲೋ|ನಮಸ್ಕಾರ|ನಮಸ್ತೆ)\b',
            r'\b(नमस्ते|नमस्कार|हैलो)\b',
            r'\b(வணக்கம்|ஹலோ)\b',
            r'\b(నమస్కారం|హలో)\b'
        ],
        'information_request': [
            r'\b(what|when|where|which|whose|how)\b',
            r'\b(ಏನು|ಯಾವಾಗ|ಎಲ್ಲಿ|ಹೇಗೆ)\b',
            r'\b(क्या|कब|कहाँ|कैसे)\b',
            r'\b(என்ன|எப்போது|எங்கே|எப்படி)\b',
            r'\b(ఏమి|ఎప్పుడు|ఎక్కడ|ఎలా)\b',
            r'\b(weather|time|date|temperature)\b',
            r'\b(ಹವಾಮಾನ|ಸಮಯ|ತಾಪಮಾನ)\b',
            r'\b(मौसम|समय|तापमान)\b'
        ],
        'task_command': [
            r'\b(open|close|start|stop|run|execute)\b',
            r'\b(ತೆರೆ|ಮುಚ್ಚು|ಪ್ರಾರಂಭ|ನಿಲ್ಲಿಸು)\b',
            r'\b(खोलो|बंद|शुरू|रोको)\b',
            r'\b(திற|மூடு|தொடங்கு|நிறுத்து)\b',
            r'\b(తెరువు|మూసివేయి|ప్రారంభించు|ఆపు)\b'
        ],
        'general_question': [
            r'\b(tell me|explain|describe|define)\b',
            r'\b(ಹೇಳು|ವಿವರಿಸು|ವಿವರಣೆ)\b',
            r'\b(बताओ|समझाओ|व्याख्या)\b',
            r'\b(சொல்|விளக்கு)\b',
            r'\b(చెప్పు|వివరించు)\b'
        ]
    }
    
    def __init__(self):
        """Initialize intent classifier."""
        print("Intent classifier initialized")
    
    def classify(self, text):
        """
        Classify intent from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected intent
        """
        text_lower = text.lower()
        
        # Check each intent pattern
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    print(f"Detected intent: {intent}")
                    return intent
        
        # Default intent
        print("Detected intent: general_question")
        return 'general_question'
