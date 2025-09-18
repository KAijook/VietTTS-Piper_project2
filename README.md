# VietTTS-document-to-speech-tool-project

Flask application for text-to-speech with voice cloning (Vietnamese) using Coqui TTS (XTTS-v2).

## Setup
1. Clone repo: `git clone https://github.com/yourusername/tts-flask-app.git`
2. Deploy to Render (see below).

## Render Deployment
- Environment: Python 3.10.12
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app`
- Env vars: `SECRET_KEY` (auto-generated)
