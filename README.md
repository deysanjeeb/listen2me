# 🎙️ Just Say The Word! - Your AI Voice Buddy

Ever wished you could just talk to your computer and get clever responses back? Well, now you can! This nifty little Python app turns your Caps Lock key into a magic button that listens to your voice and responds with AI-powered wisdom. It's like having a super-smart friend who's always ready to chat!

## ✨ What's Cool About It?

- 🎤 Talk by just holding Caps Lock (yes, that key finally has a purpose!)
- 🧠 Uses Whisper AI to understand what you're saying
- 🤖 Gemini AI crafts thoughtful responses
- 🎯 Smart enough to ignore background noise
- ⚡ Quick as a flash - just press, speak, and release!

## 🛠️ Getting Started

### What You'll Need

- Python 3.x (the newer, the better!)
- A microphone (built-in or external)
- A Google Cloud account with Gemini powers
- A quiet(ish) room (your AI buddy isn't a fan of parties)

### Setting Up Your Magic Box

1. Install the goodies:
   ```bash
   pip install numpy whisper sounddevice google-generativeai python-dotenv pynput scipy
   ```

2. Create a secret `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_super_secret_key_here
   ```

## 🎮 How to Use It

1. Fire it up:
   ```bash
   python main.py
   ```

2. The Magic Commands:
   - 🔘 Hold Caps Lock → Start talking
   - 🔄 Release Caps Lock → Get your answer
   - 🚪 Ctrl+C → Say goodbye (for now)

## 🎛️ Under the Hood

Want to tinker? Here's what you can adjust:

```python
transcriber = AudioTranscriber(
    model_type="small",        # From 'tiny' to 'large' (like coffee sizes!)
    chunk_size=1024,          # For the tech-savvy souls
    channels=1,               # Mono is all we need
    noise_threshold=0.02,     # How loud you need to be
    calibration_duration=1.0  # Getting to know your room's vibe
)
```

## 🧰 The Cool Parts

### 🎤 AudioTranscriber
The ears of the operation! It:
- Learns what your room sounds like
- Knows when you're actually talking
- Turns your beautiful voice into text

### ⌨️ CapsLockListener
The button master! It:
- Watches for your Caps Lock signal
- Starts and stops the recording
- Never misses a keystroke

## 🆘 Help! Something's Not Right!

If things go wonky:
- Check if your mic is plugged in (we've all been there)
- Try adjusting the noise threshold if it's too chatty or too shy
- Make sure your internet is awake and working

## 📝 Pro Tips

- Find a quiet spot (your AI buddy appreciates it)
- Speak clearly (no need to shout though!)
- Keep your questions clear and concise
- Have fun experimenting with different types of questions

## 🚪 Time to Go?

Just hit Ctrl+C when you're done chatting. Your AI buddy will be here when you return!

---

Made with 💖 and a bit of coding magic. Happy chatting! 🚀