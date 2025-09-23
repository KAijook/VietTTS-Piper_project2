
## 🎶 Vietnamese TTS with Piper + Gradio

A high-quality Vietnamese Text-to-Speech (TTS) application using Piper
 and a Gradio interface.
👉 Enter Vietnamese text and generate a natural .wav audio file with post-processing (noise reduction, compressor, reverb).

# You can try my model here: https://huggingface.co/spaces/Kaijook/Document-to-speech-VietTTS
# 🚀 Features

✅ Supports Piper ONNX model (vi_VN-25hours_single-low.onnx).

✅ Gradio interface with multiple controls: speed, expressiveness, stability, silence.

✅ Audio post-processing:

Noise reduction (remove background noise).

Compressor (balance volume).

Light reverb (more natural sound).

✅ Saves text → audio history for reuse.

✅ Modern pastel dark theme UI with a blinking ✨ header.

# 📦 Installation

Requirements: Python 3.9+

Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

# 🎤 Usage

Enter Vietnamese text into the input box.

Adjust the parameters as desired:

Speed: Lower value = faster speech

Expressiveness: Higher value = more expressive

Stability: Higher value = more stable

Silence: Pause duration between sentences

Click "Generate Voice".

Listen to the result and download if needed.

# 🧠 Model

Uses the vi_VN-25hours_single-low model trained on 25 hours of Vietnamese data.

# 📜 License

Not for commercial use.


