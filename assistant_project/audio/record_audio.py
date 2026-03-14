"""
Audio recording functionality.
"""
import sounddevice as sd
import soundfile as sf
import numpy as np

class AudioRecorder:
    """Records audio from microphone."""
    
    def __init__(self, sample_rate=16000):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate (int): Sample rate for recording (default: 16000 for Whisper)
        """
        self.sample_rate = sample_rate
        self.recording = None
    
    def record(self, duration=5):
        """
        Record audio for specified duration.
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        print(f"Recording for {duration} seconds...")
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Recording complete")
        return self.recording.flatten()
    
    def save(self, audio_data, filename="recording.wav"):
        """
        Save audio data to file.
        
        Args:
            audio_data (numpy.ndarray): Audio data
            filename (str): Output filename
        """
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Audio saved to {filename}")
        return filename
