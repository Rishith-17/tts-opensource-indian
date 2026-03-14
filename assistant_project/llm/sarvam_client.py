"""
Sarvam AI client for chat completions.
Excellent support for Indian languages!
"""
import re
import os
from sarvamai import SarvamAI

def _load_api_key(provided_key: str = None) -> str:
    """Resolve API key: argument → SARVAM_API_KEY env var → .env file."""
    if provided_key:
        return provided_key
    # Try environment variable
    key = os.environ.get("SARVAM_API_KEY", "")
    if key:
        return key
    # Try .env file (simple parse, no dependency on python-dotenv)
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    env_path = os.path.abspath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("SARVAM_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError(
        "Sarvam API key not found.\n"
        "Set it in .env file: SARVAM_API_KEY=your_key\n"
        "Or as environment variable: set SARVAM_API_KEY=your_key"
    )


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS doesn't read symbols aloud."""
    # Remove headers (## Heading → Heading)
    text = re.sub(r'#{1,6}\s*', '', text)
    # Remove bold/italic (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    # Remove bullet points (- item, * item, • item)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered lists (1. item → item)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove inline code (`code`)
    text = re.sub(r'`[^`]*`', '', text)
    # Remove code blocks (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove horizontal rules (---, ***)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class SarvamClient:
    """Client for Sarvam AI chat API."""
    
    def __init__(self, api_key=None, model="sarvam-30b"):
        self.api_key = _load_api_key(api_key)
        self.model = model
        self.client = SarvamAI(api_subscription_key=self.api_key)
        
        self.system_prompt = (
            "You are a helpful AI assistant. CRITICAL RULES:\n"
            "1. ALWAYS respond in the EXACT SAME LANGUAGE as the user's question\n"
            "2. If user writes in Kannada script (ಕನ್ನಡ), respond ONLY in Kannada script\n"
            "3. If user writes in Hindi script (हिंदी), respond ONLY in Hindi script\n"
            "4. If user writes in English, respond ONLY in English\n"
            "5. Do NOT translate or switch languages\n"
            "6. Do NOT use thinking mode or explain your process\n"
            "7. Give direct, helpful answers immediately\n"
            "8. Do NOT use markdown formatting — no **, no ##, no bullet points, no numbered lists. "
            "Write in plain sentences only, because your response will be spoken aloud by a TTS system."
        )
        # Initialize with system prompt for multilingual support
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        print(f"✓ Sarvam AI client initialized with model: {model}")
        print("  Supports: Kannada, Hindi, Tamil, Telugu, English + more!")
    
    def generate(self, prompt, intent="general", language="en"):
        """
        Generate response using Sarvam AI.
        
        Args:
            prompt (str): User prompt in native language
            intent (str): Intent classification (not used, for compatibility)
            language (str): Language code (kn, hi, en, ta, te)
            
        Returns:
            str: Generated response in same language as prompt
        """
        # Language name mapping
        lang_names = {
            'kn': 'Kannada',
            'hi': 'Hindi',
            'en': 'English',
            'ta': 'Tamil',
            'te': 'Telugu'
        }
        lang_name = lang_names.get(language, 'the user\'s language')
        
        try:
            # Add language-specific instruction before user message
            language_instruction = f"[Respond in {lang_name} only]"
            full_prompt = f"{language_instruction}\n{prompt}"
            
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": full_prompt
            })
            
            # Make API call using SDK
            print(f"Sending request to Sarvam AI ({self.model})...")
            print(f"Language: {lang_name} ({language})")
            print(f"Prompt: {prompt[:100]}...")
            
            response = self.client.chat.completions(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=2000,  # Increased to 2000 for longer responses
                top_p=0.9,  # Add top_p for better quality
                reasoning_effort="low"  # Disable thinking mode for direct responses
            )
            
            print(f"Response object: {response}")
            print(f"Response type: {type(response)}")
            
            # Check if response is valid
            if response is None:
                print("✗ Response is None!")
                return "Sorry, I received no response from the API. Please check your API key."
            
            if not hasattr(response, 'choices') or not response.choices:
                print(f"✗ Response has no choices: {response}")
                return "Sorry, I received an invalid response from the API."
            
            # Extract assistant message
            # Prioritize 'content' field (direct response)
            message_obj = response.choices[0].message
            assistant_message = message_obj.content
            
            # If content is None, try reasoning_content as fallback
            if assistant_message is None:
                if hasattr(message_obj, 'reasoning_content') and message_obj.reasoning_content:
                    # Model entered thinking mode despite reasoning_effort="low"
                    # This shouldn't happen often, but when it does, we need to handle it
                    reasoning = message_obj.reasoning_content
                    print("⚠ Warning: Model entered thinking mode despite reasoning_effort='low'")
                    print(f"⚠ This may indicate the query is too complex or max_tokens is too low")
                    
                    # The reasoning_content contains the thinking process, not the actual answer
                    # We should ask the user to retry or simplify the query
                    assistant_message = "I apologize, but I need to think more about this. Please try asking your question again, or try making it simpler."
                else:
                    print("✗ Both content and reasoning_content are None!")
                    return "Sorry, I received an empty response from the API."
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Keep only last 10 messages to avoid context overflow
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Strip markdown so TTS doesn't read ** ## - symbols aloud
            assistant_message = strip_markdown(assistant_message)

            print(f"✓ Sarvam response: {assistant_message[:100]}...")
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Check if it's an API key issue
            if "401" in str(e) or "403" in str(e) or "Unauthorized" in str(e):
                return "Sorry, there's an issue with the API key. Please verify it's correct."
            
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history but keep system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("✓ Conversation history cleared")
