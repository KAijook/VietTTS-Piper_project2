import sys
import types

try:
    import websockets.asyncio
except ModuleNotFoundError:
    import websockets
    sys.modules["websockets.asyncio"] = types.ModuleType("websockets.asyncio")
    sys.modules["websockets.asyncio"].__dict__.update(websockets.__dict__)

import gradio as gr
import tempfile
import requests
import os
import subprocess
import tarfile
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile



MODEL_PATH = "vi_VN-vais1000-medium.onnx"   
PIPER_TAR  = "piper_linux_x86_64.tar.gz"      
PIPER_DIR  = "./piper_bin"
PIPER_BIN  = os.path.join(PIPER_DIR, "piper")


def safe_extract(tar, path=".", members=None):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.commonprefix([os.path.abspath(path), os.path.abspath(member_path)]) == os.path.abspath(path):
            raise Exception("🚨 Path Traversal detected in tar file!")
    tar.extractall(path, members)

def setup_piper():
    if not os.path.exists(PIPER_DIR):
        os.makedirs(PIPER_DIR, exist_ok=True)

    local_piper = None
    if not os.path.exists(PIPER_BIN):
        if not os.path.exists(PIPER_TAR):
            raise RuntimeError(f"❌ Không tìm thấy {PIPER_TAR}, hãy upload vào repo Hugging Face!")

        print("📦 Đang giải nén Piper...")
        with tarfile.open(PIPER_TAR, "r:gz") as tar:
            safe_extract(tar, PIPER_DIR)
        print("✅ Giải nén Piper xong!")

    for root, dirs, files in os.walk(PIPER_DIR):
        if "piper" in files:
            local_piper = os.path.join(root, "piper")
            break

    if not local_piper:
        raise RuntimeError("❌ Không tìm thấy file 'piper' sau khi giải nén!")

    os.chmod(local_piper, 0o755)
    os.environ["LD_LIBRARY_PATH"] = f"{os.path.dirname(local_piper)}:{os.environ.get('LD_LIBRARY_PATH','')}"

    print(f"✅ Piper đã sẵn sàng tại {local_piper}")
    return local_piper


PIPER_BIN = setup_piper()

# --- Postprocess audio ---
def postprocess_audio(input_path, output_path):
    # 1. Đọc file WAV
    data, sr = sf.read(input_path)

    # 2. Khử nhiễu
    reduced = nr.reduce_noise(y=data, sr=sr)
    sf.write("tmp_clean.wav", reduced, sr)

    # 3. Nén âm + cân bằng
    audio = AudioSegment.from_wav("tmp_clean.wav")
    compressed = effects.compress_dynamic_range(audio)
    compressed.export("tmp_compressed.wav", format="wav")

    # 4. Thêm reverb nhẹ
    with AudioFile("tmp_compressed.wav") as f:
        audio_np = f.read(f.frames)
        sr = f.samplerate

    board = Pedalboard([Reverb(room_size=0.2, wet_level=0.15, dry_level=0.85)])
    effected = board(audio_np, sr)

    with AudioFile(output_path, "w", sr, effected.shape[0]) as f:
        f.write(effected)


def text_to_speech(text, length_scale=0.7, noise_scale=0.8, noise_w=0.8, sentence_silence=0.5):
    try:
        if not os.path.exists(PIPER_BIN):
            return None, "❌ Piper chưa sẵn sàng"
        if not os.path.exists(MODEL_PATH):
            return None, f"❌ Không tìm thấy model tại {MODEL_PATH}"

        tmp_output = "output.wav"
        cmd = [
            PIPER_BIN,
            "--model", MODEL_PATH,
            "--output_file", tmp_output,
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
            "--noise_w", str(noise_w),
            "--sentence_silence", str(sentence_silence)
        ]
        result = subprocess.run(cmd, input=text.encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            return None, f"❌ Lỗi Piper:\n{result.stderr.decode('utf-8')}"
        if not os.path.exists(tmp_output):
            return None, "❌ Piper không tạo ra file âm thanh!"

        # 🔊 Hậu xử lý audio
        final_output = "output_final.wav"
        postprocess_audio(tmp_output, final_output)

        return final_output, "✅ Thành công!"
    except Exception as e:
        return None, f"❌ Lỗi hệ thống: {e}"


if not os.path.exists("audios"):
    os.makedirs("audios")

history = []

def tts_and_save(text, length_scale=0.7, noise_scale=0.8, noise_w=0.8, sentence_silence=0.5):
    audio_path, msg = text_to_speech(text, length_scale, noise_scale, noise_w, sentence_silence)
    if audio_path is None:
        return msg, None, list(reversed(history))

   
    filename = f"audios/{len(history)+1}.wav"
    os.rename(audio_path, filename)

   
    history.append((text, filename))

    return msg, filename, list(reversed(history))

# ---- Gradio UI ----
custom_css = """
/* Phông nền */
.gradio-container {
    background: linear-gradient(135deg, #1c1c1c, #243b55, #141e30);
    color: #d1f7ff;
}


/* Header chính */
#header {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    background: linear-gradient(90deg, #8ec5fc, #e0c3fc, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


/* Button chính */
#submit-btn { 
    background: linear-gradient(90deg,#667eea,#764ba2); 
    color:white; 
    font-weight:bold; 
    border-radius:12px; 
    border: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    padding: 10px 20px;
}
#submit-btn:hover {
    opacity: 0.95;
    transform: scale(1.03);
    transition: all 0.2s ease;
}

/* Khung input/output */
.gr-textbox, .gr-slider, .gr-text-input, textarea, input { 
    background: rgba(255,255,255,0.85) !important; 
    color: #222 !important; 
    border-radius: 10px !important; 
    border: 1px solid #d1c4e9 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
}

/* Nhãn label (ví dụ: Tốc độ nói, Độ biểu cảm, ...) */
label { 
    font-weight: bold; 
    color: #4B0082 !important; 
}

/* Output text */
#output_text { 
    color: #333; 
    font-weight: bold; 
    background: rgba(255,255,255,0.85); 
    border-radius: 10px; 
    padding: 8px;
}

/* Dataframe lịch sử */
.gr-dataframe { 
    background: rgba(255,255,255,0.9) !important; 
    color: #222 !important; 
    border-radius: 10px !important;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

"""


with gr.Blocks(title="🎤 Vietnamese Piper TTS", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align:center;color:#4B0082;'>🎶 Vietnamese TTS (Piper)</h1>", elem_id="header")

    with gr.Tabs():
        # ---- Tab TTS ----
        with gr.TabItem("TTS"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        text_input = gr.Textbox(label="Nhập văn bản", placeholder="Nhập tiếng Việt...", lines=5)
                        length_scale = gr.Slider(0.1, 2.0, value=0.7, label="Tốc độ nói")
                        noise_scale = gr.Slider(0.1, 1.5, value=0.8, label="Độ biểu cảm")
                        noise_w = gr.Slider(0.1, 1.5, value=0.8, label="Độ ổn định")
                        sentence_silence = gr.Slider(0.0, 2.0, value=0.5, label="Khoảng lặng")
                        submit_btn = gr.Button("🎧 Tạo âm thanh", elem_id="submit-btn")
                with gr.Column(scale=1):
                    with gr.Group():
                        output_text = gr.Textbox(label="Thông báo")
                        audio_output = gr.Audio(label="Audio", type="filepath")

            history_table = gr.Dataframe(headers=["Text", "Download Audio"], datatype=["str","file"], row_count=(1,10))
            submit_btn.click(
                tts_and_save, 
                inputs=[text_input, length_scale, noise_scale, noise_w, sentence_silence], 
                outputs=[output_text, audio_output, history_table]
            )

        # ---- Tab History ----
        with gr.TabItem("History"):
            gr.Markdown("⚡ Lịch sử text → audio (mới nhất lên trên)")
            history_table2 = gr.Dataframe(headers=["Text", "Download Audio"], datatype=["str","file"], row_count=(1,10))
            submit_btn.click(lambda text, ls, ns, nw, ss: list(reversed(history)), 
                             inputs=[text_input, length_scale, noise_scale, noise_w, sentence_silence], 
                             outputs=history_table2)

 
if __name__ == "__main__":
    demo.launch()
