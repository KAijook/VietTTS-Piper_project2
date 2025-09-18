from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import os
import uuid
import re
import numpy as np
from scipy.io import wavfile
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import docx
from num2words import num2words
import logging
import time
import threading
from functools import wraps
from werkzeug.utils import secure_filename
import soundfile as sf

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Đặt biến môi trường để bỏ qua license prompt
os.environ["COQUI_TOS_AGREED"] = "y"

# Monkey-patch hàm ask_tos trong ModelManager
from TTS.utils.manage import ModelManager

def patched_ask_tos(self, output_path):
    logging.debug("Bỏ qua license prompt với patched_ask_tos")
    return True

ModelManager.ask_tos = patched_ask_tos

# Thêm XttsConfig vào danh sách an toàn của torch.load
torch.serialization.add_safe_globals([XttsConfig])

# Monkey-patch torch.load để dùng weights_only=True
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = True
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')  # Sử dụng env var cho production

# Thư mục lưu giọng mặc định (persistent trên Render qua disk mounts nếu cần)
VOICE_DIR = 'static/voices'
os.makedirs(VOICE_DIR, exist_ok=True)
DEFAULT_VOICE_PATH = os.path.join(VOICE_DIR, 'default_speaker.wav')

MODELS = {
    "vi-vn": {
        "name": "Vietnamese (XTTS-v2 với voice cloning)",
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "speakers": ["male_vi_1", "female_vi_1", "female_vi_2"]
    },
}

loaded_models = {}

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if exception[0]:
                raise exception[0]
            return result[0]

        return wrapper
    return decorator

@timeout(600)
def load_model(lang_code):
    """Load model TTS trên CPU"""
    if lang_code not in loaded_models:
        try:
            logging.debug(f"Bắt đầu tải mô hình cho {lang_code}")
            start_time = time.time()
            device = "cpu"  # Buộc CPU trên Render
            logging.debug(f"Chạy trên {device}")

            cache_dir = os.path.expanduser("~/.local/share/tts")
            model_path = os.path.join(cache_dir, MODELS[lang_code]["model"].replace("/", "--"))
            if os.path.exists(model_path):
                logging.debug(f"Cache mô hình tồn tại tại {model_path}, sử dụng cache")
                model = TTS(
                    model_name=MODELS[lang_code]["model"],
                    model_path=model_path,
                    progress_bar=False,  # Tắt progress bar để tránh log spam
                    gpu=False
                )
            else:
                logging.debug(f"Không tìm thấy cache tại {model_path}, tải từ Hugging Face")
                model = TTS(
                    model_name=MODELS[lang_code]["model"],
                    progress_bar=False,
                    gpu=False
                )

            logging.debug(f"Ngôn ngữ hỗ trợ: {model.languages}")
            if lang_code not in model.languages:
                logging.error(f"Ngôn ngữ {lang_code} không được hỗ trợ bởi mô hình")
                raise ValueError(f"Ngôn ngữ {lang_code} không được hỗ trợ. Ngôn ngữ khả dụng: {model.languages}")
            loaded_models[lang_code] = model
            elapsed_time = time.time() - start_time
            logging.debug(f"Đã tải mô hình cho {lang_code} trên {device} trong {elapsed_time:.2f} giây")
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình cho {lang_code}: {str(e)}")
            raise ValueError(f"Failed to load model for {lang_code}: {str(e)}")
    return loaded_models[lang_code]

# Các hàm khác giữ nguyên: normalize_numbers, extract_text, split_text, validate_wav, synthesize_parts
# (Copy từ phiên bản trước để đầy đủ, nhưng tôi rút gọn ở đây để ngắn gọn)

def normalize_numbers(text, lang="vi-vn"):
    # ... (giữ nguyên từ trước)
    pass

def extract_text(file):
    # ... (giữ nguyên)
    pass

def split_text(text, max_len=200):
    # ... (giữ nguyên)
    pass

def validate_wav(file_path):
    # ... (giữ nguyên)
    pass

def synthesize_parts(model, parts, base_path, lang, speaker_wav=None):
    # ... (giữ nguyên)
    pass

# Routes giữ nguyên: serve_static, upload_voice, index
# (Copy từ phiên bản trước)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_file(os.path.join("static", filename))

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    # ... (giữ nguyên)
    pass

@app.route("/", methods=["GET", "POST"])
def index():
    # ... (giữ nguyên)
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
