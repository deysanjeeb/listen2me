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

from pynput import keyboard
import threading
import time


class CapsLockListener:
    def __init__(self, callback_press, callback_release, combo_keys=None):
        """
        Initialize the Caps Lock listener

        Args:
            callback (function): Function to call when Caps Lock is pressed
            combo_keys (list): Optional additional keys required with Caps Lock
        """
        self.callback_press = callback_press
        self.callback_release = callback_release
        self.combo_keys = set(combo_keys) if combo_keys else set()
        self.current_keys = set()
        self.listener = None
        self.running = False
        self.caps_lock_pressed = False

    def on_press(self, key):
        """Handle key press events"""
        try:
            # Convert key to string representation
            key_str = key.char
        except AttributeError:
            key_str = str(key).replace("Key.", "").lower()

        self.current_keys.add(key_str)

        # Check specifically for Caps Lock
        if key_str == "caps_lock":
            self.caps_lock_pressed = True

            # If no combo keys are required, trigger immediately
            if not self.combo_keys:
                self.callback_press()
            # If combo keys are required, check if they're all pressed
            elif all(k in self.current_keys for k in self.combo_keys):
                self.callback_press()

    def on_release(self, key):
        """Handle key release events"""
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key).replace("Key.", "").lower()

        if key_str == "caps_lock":
            self.caps_lock_pressed = False
            self.callback_release()

        if key_str in self.current_keys:
            self.current_keys.remove(key_str)

    def start(self):
        """Start listening for Caps Lock"""
        self.running = True
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def stop(self):
        """Stop listening"""
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None


class AudioTranscriber:
    def __init__(
        self,
        model_type: str = "base",
        chunk_size: int = 1024,
        channels: int = 1,
        noise_threshold: float = 0.01,
        calibration_duration: float = 1.0,
    ):
        """
        Initialize the audio transcriber

        Args:
            model_type: Whisper model type ('tiny', 'base', 'small', 'medium', 'large')
            chunk_size: Size of audio chunks to process
            channels: Number of audio channels (1 for mono, 2 for stereo)
            noise_threshold: Minimum amplitude to consider as speech
            calibration_duration: Duration in seconds to calibrate noise level
        """
        self.model = whisper.load_model(model_type)
        self.chunk_size = chunk_size
        self.channels = channels
        self.noise_threshold = noise_threshold
        self.calibration_duration = calibration_duration
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.ambient_noise_level = None
        self.complete_audio_buffer = []

        # Get default device info
        device_info = sd.query_devices(kind="input")
        self.sample_rate = int(device_info["default_samplerate"])
        print(f"Using device: {device_info['name']}")
        print(f"Sample rate: {self.sample_rate}")

    def calibrate_noise(self) -> None:
        """Calibrate the ambient noise level"""
        print("Calibrating ambient noise level... Please remain quiet.")

        calibration_samples = []
        duration = 0

        def callback(indata, frames, time, status):
            if status:
                print(status)
            calibration_samples.append(np.abs(indata).mean())

        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=callback,
        ):
            time.sleep(self.calibration_duration)

        samples = np.array(calibration_samples)
        self.ambient_noise_level = 0.002
        print(f"Ambient noise level calibrated: {self.ambient_noise_level:.6f}")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Determine if audio chunk contains speech
        """
        amplitude = np.abs(audio_chunk).mean()
        return amplitude > (self.ambient_noise_level * (1 + self.noise_threshold))

    def start_recording(self) -> None:
        """Start recording from microphone"""
        # Reset the audio buffer
        self.complete_audio_buffer = []

        # Calibrate noise level first
        self.calibrate_noise()

        self.is_running = True

        def callback(indata, frames, time, status):
            """Callback for sounddevice"""
            if status:
                print(status)
            # Only store chunks that contain speech
            if self.is_speech(indata):
                self.complete_audio_buffer.append(indata.copy())

        # Start the recording stream
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=callback,
        )

        self.stream.start()
        print("Started recording... Press Ctrl+C to stop")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        from scipy import signal

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        return signal.resample(audio, target_length)

    def stop_recording(self) -> str:
        """Stop recording, process the complete audio, and return the transcription"""
        self.is_running = False

        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        print("\nProcessing complete recording...")

        # Combine all audio chunks
        if not self.complete_audio_buffer:
            print("No speech detected during recording")
            return ""

        # Concatenate all audio chunks
        complete_audio = np.concatenate(self.complete_audio_buffer)

        # Flatten if stereo
        if complete_audio.ndim > 1:
            complete_audio = complete_audio.flatten()

        # Normalize audio
        audio_normalized = (
            complete_audio / np.max(np.abs(complete_audio))
            if np.max(np.abs(complete_audio)) > 0
            else complete_audio
        )

        # Resample to 16kHz for Whisper if needed
        if self.sample_rate != 16000:
            audio_resampled = self._resample(audio_normalized, self.sample_rate, 16000)
        else:
            audio_resampled = audio_normalized

        # Transcribe
        try:
            result = self.model.transcribe(audio_resampled, language="en", fp16=False)
            transcription = result["text"].strip()
            print(f"Complete Transcription: {transcription}")
            return transcription
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""


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
    list_audio_devices()

    transcriber = AudioTranscriber(
        model_type="small",
        chunk_size=1024,
        channels=1,
        noise_threshold=0.02,  # Adjust this value based on your environment
        calibration_duration=1.0,
    )

    def on_caps_lock():
        transcriber.start_recording()

    def off_caps_lock():
        transcriber.stop_recording()

    listener = CapsLockListener(
        callback_press=on_caps_lock, callback_release=off_caps_lock
    )

    try:
        print("Listening for Caps Lock...")
        print("Press ctrl + c to exit")
        listener.start()

        while listener.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping listener...")
        listener.stop()

    # try:
    #     # Show available devices

    #     transcriber.start_recording()

    #     # Keep running until Ctrl+C
    #     try:
    #         while True:
    #             time.sleep(0.1)
    #     except KeyboardInterrupt:
    #         transcriber.stop_recording()

    # except Exception as e:
    #     print(f"Error: {e}")


if __name__ == "__main__":
    main()
