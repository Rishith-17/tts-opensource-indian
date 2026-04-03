"""
Core TTS functionality using Coqui TTS with pretrained models.
"""
import os
import torch
import json
import numpy as np
from TTS.utils.synthesizer import Synthesizer
from language_router import LanguageRouter

class MultilingualTTS:
    """Multilingual Text-to-Speech system."""
    
    def __init__(self):
        """Initialize TTS system."""
        self.router = LanguageRouter()
        self.synthesizers = {}  # Cache synthesizers per language
        self.current_language = None
        # Force CPU for TTS to avoid device mismatch issues with multi-speaker models
        # Ollama will still use GPU for LLM inference
        self.device = "cpu"
        print(f"Using device: {self.device}")
    
    def load_model(self, language_code):
        """
        Load TTS model for specified language.
        
        Args:
            language_code (str): Language code (kn, hi, en, ta, te)
            
        Returns:
            Synthesizer: Loaded TTS synthesizer
        """
        # Return cached if already loaded
        if language_code in self.synthesizers:
            self.current_language = language_code
            return self.synthesizers[language_code]
        
        model_path = self.router.get_model_path(language_code)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please ensure the model files are in the correct location."
            )
        
        print(f"Loading {language_code} model from {model_path}...")
        
        try:
            # Paths to model files
            tts_checkpoint = os.path.join(model_path, "fastpitch", "best_model.pth")
            tts_config_path = os.path.join(model_path, "fastpitch", "config.json")
            vocoder_checkpoint = os.path.join(model_path, "hifigan", "best_model.pth")
            vocoder_config_path = os.path.join(model_path, "hifigan", "config.json")
            
            # Verify TTS files exist
            if not os.path.exists(tts_checkpoint):
                raise FileNotFoundError(f"TTS checkpoint not found: {tts_checkpoint}")
            if not os.path.exists(tts_config_path):
                raise FileNotFoundError(f"TTS config not found: {tts_config_path}")
            
            # Fix config paths
            with open(tts_config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Handle speakers file - use absolute path
            speakers_file = os.path.join(model_path, "fastpitch", "speakers.pth")
            if os.path.exists(speakers_file):
                # Convert to absolute path
                speakers_file = os.path.abspath(speakers_file)
                config_dict['speakers_file'] = speakers_file
                if 'model_args' in config_dict:
                    config_dict['model_args']['speakers_file'] = speakers_file
            else:
                # Disable speaker embedding if no speakers file
                config_dict['speakers_file'] = None
                config_dict['use_speaker_embedding'] = False
                config_dict['num_speakers'] = 0
                if 'model_args' in config_dict:
                    config_dict['model_args']['speakers_file'] = None
                    config_dict['model_args']['use_speaker_embedding'] = False
                    config_dict['model_args']['num_speakers'] = 0
            
            # Save fixed config
            temp_config_path = os.path.join(model_path, "fastpitch", "config_fixed.json")
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
            # Check vocoder availability
            vocoder_path = None
            vocoder_config = None
            if os.path.exists(vocoder_checkpoint) and os.path.exists(vocoder_config_path):
                vocoder_path = vocoder_checkpoint
                vocoder_config = vocoder_config_path
                print(f"✓ Using HiFiGAN vocoder")
            else:
                print(f"⚠ Vocoder not found, will use Griffin-Lim (slower)")
            
            # Initialize Synthesizer (use CPU to avoid device mismatch with multi-speaker models)
            # Note: Ollama LLM will still use GPU for fast inference
            synthesizer = Synthesizer(
                tts_checkpoint=tts_checkpoint,
                tts_config_path=temp_config_path,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config,
                use_cuda=False  # Force CPU for stability with multi-speaker models
            )
            
            self.synthesizers[language_code] = synthesizer
            self.current_language = language_code
            print(f"✓ Model loaded successfully for {language_code}")
            return synthesizer
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_speech(self, text, language_code=None, speaker_name="female"):
        """
        Generate speech from text.
        
        Args:
            text (str): Input text
            language_code (str, optional): Language code. Auto-detected if None.
            speaker_name (str, optional): Speaker voice options:
                - "female"        : Default female voice
                - "male"          : Default male voice
                - "female_slow"   : Female, slower pace (clearer)
                - "male_slow"     : Male, slower pace (clearer)
                - "female_fast"   : Female, faster pace
                - "male_fast"     : Male, faster pace
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        # Detect language if not provided
        if language_code is None:
            language_code = self.router.detect_language(text)
            print(f"Detected language: {language_code}")
        
        # Parse speaker name and speed modifier
        speed_scale = 1.0  # Default speed
        if speaker_name.endswith("_slow"):
            speed_scale = 1.3  # 30% slower = clearer pronunciation
            speaker_name = speaker_name.replace("_slow", "")
        elif speaker_name.endswith("_fast"):
            speed_scale = 0.85  # 15% faster
            speaker_name = speaker_name.replace("_fast", "")

        # Validate speaker name
        if speaker_name not in ("female", "male"):
            speaker_name = "female"

        # Male voice: slightly slower for natural, professional feel
        if speaker_name == "male" and speed_scale == 1.0:
            speed_scale = 1.1   # 10% slower — natural customer care pacing
        
        # Clean text - replace problematic characters
        text = text.replace("\u2019", "'")
        text = text.replace("\u2018", "'")
        text = text.replace("\u201c", '"')
        text = text.replace("\u201d", '"')
        # Remove markdown bold/italic markers
        text = text.replace("**", "").replace("*", "")
        # Remove bullet points
        text = text.replace("- ", "").replace("• ", "")
        
        # Load appropriate model
        synthesizer = self.load_model(language_code)
        
        # Generate speech
        print(f"Generating speech for: {text[:50]}...")
        print(f"Speaker: {speaker_name}, Speed scale: {speed_scale}")
        
        try:
            wav = synthesizer.tts(
                text=text,
                speaker_name=speaker_name,
                speed=speed_scale
            )
            return np.array(wav)
        except TypeError:
            # Older Coqui TTS versions don't support speed parameter
            wav = synthesizer.tts(text=text, speaker_name=speaker_name)
            return np.array(wav)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate speech: {e}")
    
    def save_audio(self, wav, output_path="output.wav"):
        import soundfile as sf

        if self.current_language in self.synthesizers:
            sample_rate = self.synthesizers[self.current_language].output_sample_rate
        else:
            sample_rate = 22050

        wav = np.array(wav, dtype=np.float32)

        # Remove DC offset only
        wav = wav - np.mean(wav)

        # Clean normalize — no filters, no clipping, no distortion
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.90

        # Fade in/out 5ms to avoid clicks at boundaries
        fade = int(sample_rate * 0.005)
        if len(wav) > fade * 2:
            wav[:fade]  *= np.linspace(0, 1, fade)
            wav[-fade:] *= np.linspace(1, 0, fade)

        sf.write(output_path, wav, sample_rate, subtype='PCM_16')
        print(f"Audio saved to: {output_path}")

