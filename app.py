from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import os
import sys
import uuid
import re
import numpy as np
from scipy.io import wavfile
import torch
from TTS.api import TTS  # Import từ coqui-tts
from TTS.tts.configs.xtts_config import XttsConfig
import docx
from num2words import num2words
import logging
import time
import threading
from functools import wraps
from werkzeug.utils import secure_filename
import soundfile as sf

# Kiểm tra Python version (phải >=3.10, <3.13 cho coqui-tts 0.27.1)
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise RuntimeError("coqui-tts requires Python >=3.10, <3.13")

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
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')

# Thư mục lưu giọng mặc định
VOICE_DIR = 'static/voices'
os.makedirs(VOICE_DIR, exist_ok=True)
DEFAULT_VOICE_PATH = os.path.join(VOICE_DIR, 'default_speaker.wav')

MODELS = {
    "vi-vn": {
        "name": "Vietnamese (XTTS-v2 với voice cloning)",
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "speakers": ["male_vi_1", "female_vi_1", "female_vi_2"],
        "fallback_model": "tts_models/vi/vits"  # Fallback nếu XTTS chậm
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
    """Load model TTS trên CPU, fallback sang VITS nếu cần"""
    if lang_code not in loaded_models:
        try:
            logging.debug(f"Bắt đầu tải mô hình cho {lang_code}")
            start_time = time.time()
            device = "cpu"
            logging.debug(f"Chạy trên {device}")

            cache_dir = os.path.expanduser("~/.local/share/tts")
            model_path = os.path.join(cache_dir, MODELS[lang_code]["model"].replace("/", "--"))
            try:
                if os.path.exists(model_path):
                    logging.debug(f"Cache mô hình tồn tại tại {model_path}, sử dụng cache")
                    model = TTS(
                        model_name=MODELS[lang_code]["model"],
                        model_path=model_path,
                        progress_bar=False,
                        gpu=False
                    )
                else:
                    logging.debug(f"Không tìm thấy cache tại {model_path}, tải XTTS-v2 từ Hugging Face")
                    model = TTS(
                        model_name=MODELS[lang_code]["model"],
                        progress_bar=False,
                        gpu=False
                    )
            except Exception as e:
                logging.warning(f"Không tải được XTTS-v2: {str(e)}. Fallback sang VITS.")
                model_path = os.path.join(cache_dir, MODELS[lang_code]["fallback_model"].replace("/", "--"))
                if os.path.exists(model_path):
                    logging.debug(f"Cache VITS tồn tại tại {model_path}")
                    model = TTS(
                        model_name=MODELS[lang_code]["fallback_model"],
                        model_path=model_path,
                        progress_bar=False,
                        gpu=False
                    )
                else:
                    logging.debug(f"Tải VITS từ Hugging Face")
                    model = TTS(
                        model_name=MODELS[lang_code]["fallback_model"],
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

def normalize_numbers(text, lang="vi-vn"):
    """Chuyển số, tiền tệ, ngày tháng, số điện thoại thành văn bản tự nhiên"""
    logging.debug("Bắt đầu chuẩn hóa văn bản")
    start_time = time.time()

    def replace_number(match):
        num_str = match.group(0)
        if re.match(r'[\d,]+\.?\d*\s*(VNĐ|\$|€|USD)', num_str):
            num_part = re.match(r'[\d,]+(\.\d+)?', num_str).group(0)
            currency = re.search(r'(VNĐ|\$|€|USD)', num_str).group(0)
            num_clean = num_part.replace(",", "")
            if "." in num_clean:
                integer, decimal = num_clean.split(".")
                integer_text = num2words(int(integer), lang=lang)
                decimal_text = " ".join(num2words(int(d), lang=lang) for d in decimal)
                return f"{integer_text} {currency.lower()} phẩy {decimal_text}"
            else:
                return f"{num2words(int(num_clean), lang=lang)} {currency.lower()}"
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', num_str):
            day, month, year = re.split(r'[/-]', num_str)
            day_text = num2words(int(day), lang=lang)
            month_text = num2words(int(month), lang=lang)
            year_text = num2words(int(year), lang=lang)
            return f"ngày {day_text} tháng {month_text} năm {year_text}"
        if re.match(r'\b\d{10}\b', num_str):
            groups = [num_str[i:i+3] if i % 6 == 0 else num_str[i:i+4] for i in range(0, len(num_str), 3)]
            groups = [g for g in groups if g]
            return " ".join(num2words(int(g), lang=lang) for g in groups)
        if re.match(r'\b\d+\.\d+\b', num_str):
            integer, decimal = num_str.split(".")
            integer_text = num2words(int(integer), lang=lang)
            decimal_text = " ".join(num2words(int(d), lang=lang) for d in decimal)
            return f"{integer_text} phẩy {decimal_text}"
        if num_str.isdigit():
            return num2words(int(num_str), lang=lang)
        return num_str

    pattern = r'[\d,]+\.?\d*\s*(VNĐ|\$|€|USD)|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\b\d{10}\b|\b\d+\.\d+\b|\b\d+\b'
    result = re.sub(pattern, replace_number, text)
    elapsed_time = time.time() - start_time
    logging.debug(f"Chuẩn hóa văn bản hoàn tất trong {elapsed_time:.2f} giây")
    return result

def extract_text(file):
    """Đọc nội dung file txt/docx"""
    logging.debug(f"Đọc file: {file.filename}")
    filename = file.filename.lower()
    try:
        if filename.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            raise ValueError("Unsupported file format (only .txt, .docx supported)")
    except Exception as e:
        logging.error(f"Lỗi khi đọc file {filename}: {str(e)}")
        raise ValueError(f"Failed to process file {filename}: {str(e)}")

def split_text(text, max_len=100):  # Giảm max_len để nhanh hơn trên CPU
    """Chia text thành các phần bằng . và \n, mỗi phần <= max_len ký tự"""
    logging.debug("Bắt đầu chia văn bản")
    start_time = time.time()
    sentences = re.split(r'(?<=[.!?])\s+|\n', text.strip())
    parts = []
    current_part = ""
    for sent in sentences:
        if len(current_part + sent) < max_len:
            current_part += sent + " "
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = sent + " "
    if current_part:
        parts.append(current_part.strip())
    elapsed_time = time.time() - start_time
    logging.debug(f"Chia văn bản thành {len(parts)} phần trong {elapsed_time:.2f} giây")
    return parts

def validate_wav(file_path):
    """Kiểm tra file WAV hợp lệ và độ dài (3-10 giây)"""
    try:
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
            if duration < 3 or duration > 10:
                raise ValueError(f"File WAV phải có độ dài từ 3 đến 10 giây, hiện tại: {duration:.2f} giây")
            if f.format != 'WAV':
                raise ValueError("File không phải định dạng WAV")
    except Exception as e:
        logging.error(f"Lỗi kiểm tra file WAV: {str(e)}")
        raise ValueError(f"File WAV không hợp lệ: {str(e)}")

def synthesize_parts(model, parts, base_path, lang, speaker_wav=None):
    """Generate audio cho từng part và merge"""
    logging.debug(f"Bắt đầu tạo audio cho {len(parts)} phần")
    start_time = time.time()
    os.makedirs("static", exist_ok=True)
    audio_files = []
    sample_rate = 24000
    for i, part in enumerate(parts):
        part_path = f"{base_path}_{i}.wav"
        logging.debug(f"Tạo audio cho phần {i + 1}/{len(parts)}")
        part_start = time.time()
        kwargs = {
            "text": part,
            "file_path": part_path,
            "language": lang
        }
        if speaker_wav and os.path.exists(speaker_wav):
            validate_wav(speaker_wav)
            kwargs["speaker_wav"] = speaker_wav
            logging.debug(f"Sử dụng custom voice từ {speaker_wav}")
        else:
            default_speaker = MODELS[lang]["speakers"][0]
            kwargs["speaker"] = default_speaker
            logging.debug(f"Sử dụng predefined speaker: {default_speaker}")

        model.tts_to_file(**kwargs)
        audio_files.append(part_path)
        logging.debug(f"Phần {i + 1} hoàn tất trong {time.time() - part_start:.2f} giây")

    logging.debug("Bắt đầu merge audio")
    merged_data = np.array([], dtype=np.int16)
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.int16)
    for path in audio_files:
        sr, data = wavfile.read(path)
        merged_data = np.concatenate([merged_data, data, silence])

    wavfile.write(base_path, sample_rate, merged_data)
    for path in audio_files:
        os.remove(path)
    elapsed_time = time.time() - start_time
    logging.debug(f"Tạo và merge audio hoàn tất trong {elapsed_time:.2f} giây")
    return base_path

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Phục vụ file tĩnh như CSS"""
    return send_file(os.path.join("static", filename))

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    """Upload và lưu file WAV làm giọng mặc định"""
    if 'voice_file' not in request.files:
        flash('Không có file được chọn.')
        return redirect(url_for('index'))

    file = request.files['voice_file']
    if file.filename == '':
        flash('Không có file được chọn.')
        return redirect(url_for('index'))

    if file and file.filename.lower().endswith('.wav'):
        try:
            temp_path = os.path.join(VOICE_DIR, secure_filename(file.filename))
            file.save(temp_path)
            validate_wav(temp_path)
            os.replace(temp_path, DEFAULT_VOICE_PATH)
            flash(f'Đã lưu giọng mặc định từ {file.filename}.')
            logging.debug(f"Đã lưu custom voice tại {DEFAULT_VOICE_PATH}")
        except Exception as e:
            flash(f'Lỗi khi lưu file: {str(e)}')
            logging.error(f"Lỗi upload voice: {str(e)}")
    else:
        flash('Chỉ hỗ trợ file .wav.')
    return redirect(url_for('index'))

@app.route("/", methods=["GET", "POST"])
def index():
    lang = "vi-vn"
    if request.method == "POST":
        if 'document' in request.files:
            file = request.files.get("document")
            if file:
                try:
                    logging.debug("Bắt đầu xử lý request POST")
                    start_time = time.time()
                    os.makedirs("static", exist_ok=True)
                    text = extract_text(file)
                    if len(text) > 5000:
                        text = text[:5000]
                        logging.warning("Văn bản dài hơn 5000 ký tự, cắt bớt")

                    text = normalize_numbers(text, lang)
                    model = load_model(lang)
                    output_path = f"static/{uuid.uuid4().hex}.wav"

                    parts = split_text(text)
                    speaker_wav = DEFAULT_VOICE_PATH if os.path.exists(DEFAULT_VOICE_PATH) else None
                    synthesize_parts(model, parts, output_path, lang, speaker_wav)

                    output_filename = f"{os.path.splitext(file.filename)[0]}_vietnamese.wav"
                    elapsed_time = time.time() - start_time
                    logging.debug(f"Xử lý POST hoàn tất trong {elapsed_time:.2f} giây")
                    return send_file(output_path, as_attachment=True, download_name=output_filename)

                except Exception as e:
                    logging.error(f"Lỗi khi xử lý POST: {str(e)}")
                    flash(str(e))
        elif 'voice_file' in request.files:
            return redirect(url_for('upload_voice'))

    has_custom_voice = os.path.exists(DEFAULT_VOICE_PATH)
    return render_template("index.html", models=MODELS, error=None, lang=lang, has_custom_voice=has_custom_voice)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
