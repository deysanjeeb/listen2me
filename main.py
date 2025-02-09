import pyaudio
import wave
import numpy as np
import whisper
import threading
import queue
import time
from typing import Optional
import warnings
import sounddevice as sd


class AudioTranscriber:
    def __init__(
        self, model_type: str = "base", chunk_size: int = 1024, channels: int = 1
    ):
        """
        Initialize the audio transcriber

        Args:
            model_type: Whisper model type ('tiny', 'base', 'small', 'medium', 'large')
            chunk_size: Size of audio chunks to process
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.model = whisper.load_model(model_type)
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_running = False

        # Get default device info
        device_info = sd.query_devices(kind="input")
        self.sample_rate = int(device_info["default_samplerate"])
        print(f"Using device: {device_info['name']}")
        print(f"Sample rate: {self.sample_rate}")

    def start_recording(self) -> None:
        """Start recording from microphone"""
        self.is_running = True

        def callback(indata, frames, time, status):
            """Callback for sounddevice"""
            if status:
                print(status)
            self.audio_queue.put(indata.copy())

        # Start the recording stream
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=callback,
        )

        self.stream.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()

        print("Started recording... Press Ctrl+C to stop")

    def _process_audio(self) -> None:
        """Process audio chunks and transcribe"""
        buffer = np.array([], dtype=np.float32)

        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1)

                # Flatten if stereo
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.flatten()

                buffer = np.append(buffer, audio_chunk)

                # Process when buffer reaches 2 seconds
                if len(buffer) >= self.sample_rate * 2:
                    # Normalize audio
                    audio_normalized = (
                        buffer / np.max(np.abs(buffer))
                        if np.max(np.abs(buffer)) > 0
                        else buffer
                    )

                    # Resample to 16kHz for Whisper if needed
                    if self.sample_rate != 16000:
                        audio_resampled = self._resample(
                            audio_normalized, self.sample_rate, 16000
                        )
                    else:
                        audio_resampled = audio_normalized

                    # Transcribe
                    try:
                        result = self.model.transcribe(
                            audio_resampled, language="en", fp16=False
                        )

                        if result["text"].strip():
                            print(f"Transcription: {result['text'].strip()}")

                    except Exception as e:
                        print(f"Transcription error: {e}")

                    # Reset buffer
                    buffer = np.array([], dtype=np.float32)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                break

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        from scipy import signal

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        return signal.resample(audio, target_length)

    def stop_recording(self) -> None:
        """Stop recording and clean up"""
        self.is_running = False

        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        if hasattr(self, "process_thread"):
            self.process_thread.join()

        print("\nStopped recording")


def list_audio_devices():
    """List all available audio input devices"""
    print("\nAvailable Audio Input Devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"Index {i}: {device['name']}")
            print(f"    Channels: {device['max_input_channels']}")
            print(f"    Sample Rate: {device['default_samplerate']}")
    print("-" * 50)


def main():
    """Example usage"""
    try:
        # Show available devices
        list_audio_devices()

        transcriber = AudioTranscriber(model_type="base", chunk_size=1024, channels=1)

        transcriber.start_recording()

        # Keep running until Ctrl+C
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            transcriber.stop_recording()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
