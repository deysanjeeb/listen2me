import os


import numpy as np
import whisper

import threading
import queue
from typing import Optional
import warnings
import sounddevice as sd
import google.generativeai as genai
from dotenv import load_dotenv
from pynput import keyboard
import time
import pyautogui

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class CapsLockListener:
    def __init__(
        self, callback_press, callback_release, callback_combo, combo_keys=None
    ):
        """
        Initialize the Caps Lock listener with various callbacks for different key actions

        Args:
            callback_press (function): Function to call when Caps Lock is pressed alone
            callback_release (function): Function to call when Caps Lock is released (no combo)
            callback_combo (function): Function to call when Caps Lock + combo keys are released
            combo_keys (list): Additional keys required with Caps Lock for combo action
        """
        self.callback_press = callback_press
        self.callback_release = callback_release
        self.callback_combo = callback_combo
        self.combo_keys = set(combo_keys) if combo_keys else set()
        self.current_keys = set()
        self.listener = None
        self.running = False
        self.caps_lock_pressed = False
        self.combo_activated = False

    def on_press(self, key):
        """Handle key press events"""
        try:
            # Convert key to string representation
            key_str = (
                key.char
                if hasattr(key, "char")
                else str(key).replace("Key.", "").lower()
            )
        except AttributeError:
            key_str = str(key).replace("Key.", "").lower()

        # Add key to currently pressed keys
        self.current_keys.add(key_str)

        # Check specifically for Caps Lock
        if key_str == "caps_lock":
            self.caps_lock_pressed = True
            # Only trigger the press callback if no combo keys are being pressed
            if not any(k in self.current_keys for k in self.combo_keys):
                self.callback_press()

        # Check if we have a combo activation
        if (
            self.caps_lock_pressed
            and self.combo_keys
            and all(k in self.current_keys for k in self.combo_keys)
        ):
            self.combo_activated = True

    def on_release(self, key):
        """Handle key release events"""
        try:
            key_str = (
                key.char
                if hasattr(key, "char")
                else str(key).replace("Key.", "").lower()
            )
        except AttributeError:
            key_str = str(key).replace("Key.", "").lower()

        # Handle Caps Lock release
        if key_str == "caps_lock":
            self.caps_lock_pressed = False

            # If combo was activated, call the combo callback
            if self.combo_activated:
                self.callback_combo()
                self.combo_activated = False
            else:
                # Otherwise, just a normal caps lock release
                self.callback_release()

        # Remove the released key from current keys
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
        # Using fixed noise level as in original code
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
        print("Started recording... Press Caps Lock to stop")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        from scipy import signal

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        return signal.resample(audio, target_length)

    def stop_recording(self) -> str:
        """Stop recording, process the complete audio, and return the transcription"""
        if not self.is_running:
            return ""

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


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite-preview-02-05",
    generation_config=generation_config,
)


def main():
    transcriber = AudioTranscriber(
        model_type="small",
        chunk_size=1024,
        channels=1,
        noise_threshold=0.02,
        calibration_duration=1.0,
    )

    chat_session = model.start_chat()

    def on_caps_lock_press():
        print("Caps Lock pressed - Starting recording...")
        transcriber.start_recording()

    def on_caps_lock_release():
        print("Caps Lock released - Processing recording...")
        transcription = transcriber.stop_recording()
        if transcription:
            response = chat_session.send_message(transcription)
            print(f"AI Response: {response.text}")

        else:
            print("No transcription available or false alarm.")

    def type_transcription():
        print("Caps Lock + A combo detected - Auto-typing transcription...")
        text = transcriber.stop_recording()
        if text:
            # Type the text character by character
            for character in text:
                pyautogui.write(character)
                time.sleep(0.01)  # Adjust typing speed as needed
            # pyautogui.write(text, interval=0.01)

            print("Text successfully typed at cursor position.")
        else:
            print("No transcription available to type.")

    # Set up the listener with proper callbacks
    listener = CapsLockListener(
        callback_press=on_caps_lock_press,
        callback_release=on_caps_lock_release,
        callback_combo=type_transcription,
        combo_keys=["shift"],
    )

    try:
        print("Starting key listener...")
        print("Press Caps Lock to start recording")
        print("Press Caps Lock + A to start recording and auto-type the transcription")

        print("Press Ctrl+C to exit")
        listener.start()

        # Keep the main thread alive
        while listener.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping listener...")
        listener.stop()
        print("Program terminated.")


if __name__ == "__main__":
    main()
