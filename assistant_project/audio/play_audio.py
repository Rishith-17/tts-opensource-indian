"""
Audio playback functionality.
"""
import os
import platform
import sounddevice as sd
import soundfile as sf

class AudioPlayer:
    """Plays audio files."""
    
    def __init__(self):
        """Initialize audio player."""
        pass
    
    def play(self, audio_path):
        """
        Play audio file.
        
        Args:
            audio_path (str): Path to audio file
        """
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
        
        try:
            # Read audio file
            data, sample_rate = sf.read(audio_path)
            
            # Play audio
            print(f"Playing audio: {audio_path}")
            sd.play(data, sample_rate)
            sd.wait()
            
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def play_system(self, audio_path):
        """
        Play audio using system default player (fallback).
        
        Args:
            audio_path (str): Path to audio file
        """
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
        
        system = platform.system()
        
        try:
            if system == "Windows":
                os.startfile(audio_path)
            elif system == "Linux":
                os.system(f"aplay {audio_path} 2>/dev/null || paplay {audio_path} 2>/dev/null")
            elif system == "Darwin":
                os.system(f"afplay {audio_path}")
            
            print(f"Playing audio: {audio_path}")
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
