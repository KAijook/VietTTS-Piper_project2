import sys
import types

try:
    # Cần cho Gradio/websockets trong một số môi trường
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
import json

# --- Story Database ---
STORY_FILE = "stories.json"
STORIES = {"Không có truyện": "Chưa tải stories.json lên!"} # Khởi tạo giá trị mặc định

if os.path.exists(STORY_FILE):
    try:
        with open(STORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                STORIES = data
    except json.JSONDecodeError:
        pass
    except Exception as e:
        pass

# --- Config ---
PIPER_TAR  = "piper_linux_x86_64.tar.gz"  # Tên file nén Piper
PIPER_DIR  = "./piper_bin"
PIPER_BIN  = os.path.join(PIPER_DIR, "piper")

MODEL_CHOICES = {
    "VIVOS X Low": "models/vi_VN-vivos-x_low.onnx",
    "25hours Single Low": "models/vi_VN-25hours_single-low.onnx",
    "VAIS1000 Medium": "models/vi_VN-vais1000-medium.onnx"
}

# --- Setup Piper ---
def safe_extract(tar, path=".", members=None):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.commonprefix([os.path.abspath(path), os.path.abspath(member_path)]) == os.path.abspath(path):
            raise Exception("Path Traversal detected in tar file!")
    tar.extractall(path, members)

def setup_piper():
    if not os.path.exists(PIPER_DIR):
        os.makedirs(PIPER_DIR, exist_ok=True)

    local_piper = None
    if not os.path.exists(PIPER_BIN):
        if not os.path.exists(PIPER_TAR):
            raise RuntimeError("Piper binary or archive missing.") 

        with tarfile.open(PIPER_TAR, "r:gz") as tar:
            safe_extract(tar, PIPER_DIR)

    for root, dirs, files in os.walk(PIPER_DIR):
        if "piper" in files:
            local_piper = os.path.join(root, "piper")
            break

    if not local_piper:
        raise RuntimeError("Piper binary not found after extraction.")

    os.chmod(local_piper, 0o755)
    os.environ["LD_LIBRARY_PATH"] = f"{os.path.dirname(local_piper)}:{os.environ.get('LD_LIBRARY_PATH','')}"

    return local_piper

PIPER_BIN = setup_piper()

# --- Postprocess audio ---
def postprocess_audio(input_path, output_path):
    data, sr = sf.read(input_path)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_clean_file:
        tmp_clean_path = tmp_clean_file.name
        
        # 1. Giảm nhiễu (Noise Reduction)
        reduced = nr.reduce_noise(y=data, sr=sr)
        sf.write(tmp_clean_path, reduced, sr)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_compressed_file:
            tmp_compressed_path = tmp_compressed_file.name
            
            # 2. Nén dải động (Dynamic Range Compression)
            audio = AudioSegment.from_wav(tmp_clean_path)
            compressed = effects.compress_dynamic_range(audio)
            compressed.export(tmp_compressed_path, format="wav")

            # 3. Thêm Reverb (hiệu ứng âm thanh)
            with AudioFile(tmp_compressed_path) as f:
                audio_np = f.read(f.frames)
                sr = f.samplerate

            board = Pedalboard([Reverb(room_size=0.2, wet_level=0.15, dry_level=0.85)])
            effected = board(audio_np, sr)

            with AudioFile(output_path, "w", sr, effected.shape[0]) as f:
                f.write(effected)

# --- TTS ---
def text_to_speech(text, model_choice, length_scale=0.7, noise_scale=0.8, noise_w=0.8, sentence_silence=0.5):
    try:
        model_path = MODEL_CHOICES[model_choice]
        final_output = "output_final.wav"

        # Kiểm tra sự tồn tại của Piper và Model
        if not os.path.exists(PIPER_BIN) or not os.path.exists(model_path):
            return None
        
        # Chạy Piper trong file tạm
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_piper_output:
            output_file = tmp_piper_output.name

            cmd = [
                PIPER_BIN,
                "--model", model_path,
                "--output_file", output_file,
                "--length_scale", str(length_scale),
                "--noise_scale", str(noise_scale),
                "--noise_w", str(noise_w),
                "--sentence_silence", str(sentence_silence)
            ]
            result = subprocess.run(cmd, input=text.encode("utf-8"),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                os.remove(output_file) if os.path.exists(output_file) else None
                return None

            if not os.path.exists(output_file):
                return None
            
            # Post-process và dọn dẹp
            postprocess_audio(output_file, final_output)
            os.remove(output_file)
            
            return final_output
    except Exception as e:
        return None
        
# --- Lưu lịch sử ---
if not os.path.exists("audios"):
    os.makedirs("audios")
history = [] # Lưu dưới dạng list of dictionaries

def tts_and_save(text, model_choice, length_scale, noise_scale, noise_w, sentence_silence):
    audio_path = text_to_speech(text, model_choice, length_scale, noise_scale, noise_w, sentence_silence)
    
    table = history

    # Nếu thất bại (audio_path là None), trả về None cho audio và history cũ
    if audio_path is None:
        return None, list(reversed(table)), list(reversed(table))

    # Nếu thành công
    if not os.path.exists("audios"):
        os.makedirs("audios")
        
    filename = f"audios/{len(history)+1}.wav"
    os.rename(audio_path, filename)
    
    history.append({
        "Text": text,
        "Model": model_choice,
        "Audio": filename
    })

    # Trả về 3 output: audio_path, table1, table2
    return filename, list(reversed(history)), list(reversed(history))
    
# --- CSS ---
custom_css = """

/* Tông màu tối, ấm áp, sang trọng */
.gradio-container {
    background: linear-gradient(135deg, #1c1c1c, #243b55, #141e30);
    color: #d1f7ff;
}
/* Header chính - Hiệu ứng Ánh sáng sao */
#header {
font-size: 28px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    /* Hiệu ứng tỏa sáng ấm áp */
    text-shadow: 0 0 10px #ff7e5f, 0 0 20px #ff7e5f;  
    color: white;  
}

/* Button chính - Nổi bật */
#submit-btn { 
    background: linear-gradient(90deg, #ff7e5f, #feb47b); /* Cam ấm */
    color: #333; 
    font-weight: 900; 
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(255,126,95,0.4); /* Shadow rực rỡ hơn */
}

/* Khung input/output */
.gr-textbox, .gr-slider, .gr-text-input, textarea, input { 
    background: rgba(255,255,255,0.05) !important; /* Gần như trong suốt trên nền tối */
    color: #fff !important; 
    border: 1px solid #ff7e5f44 !important; /* Viền mờ màu nhấn */
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}

/* Nhãn label */
label { 
    color: #feb47b !important; /* Màu nhấn ấm áp */
}

/* Output text (Giữ lại ID cho các thành phần khác sử dụng) */
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

    # --- UI ---
initial_history_table_value = [{"Text": "", "Model": "", "Audio": ""}]

with gr.Blocks(title="🎤 Vietnamese Piper TTS - Storyteller", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<h1 id='header'>📖 Kể Chuyện Ngắn với Giọng Đọc AI (Piper)</h1>", elem_id="header")

    with gr.Tabs():
        # 🟣 TAB 1: KỂ CHUYỆN & CẤU HÌNH
        with gr.TabItem("🎧 Tạo Giọng Kể Chuyện"):
            with gr.Row():
                # --- CỘT 1: NHẬP LIỆU & NỘI DUNG ---
                with gr.Column(scale=2):
                    gr.Markdown("## 📝 Nội dung Truyện")
                    mode = gr.Radio(
                        ["Chọn truyện có sẵn", "Nhập văn bản"], 
                        value="Chọn truyện có sẵn", 
                        label="Chế độ nhập liệu"
                    )
                    story_dropdown = gr.Dropdown(
                        choices=list(STORIES.keys()),
                        value=list(STORIES.keys())[0],
                        label="Chọn truyện"
                    )
                    story_preview = gr.Textbox(
                        label="Nội dung truyện (có thể chỉnh sửa)",
                        lines=12,
                        interactive=True
                    )
                    text_input = gr.Textbox(
                        label="Nhập văn bản tùy ý", 
                        lines=12,
                        visible=False
                    )

                # --- CỘT 2: CẤU HÌNH GIỌNG NÓI & OUTPUT ---
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ Cấu hình Giọng nói")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_CHOICES.keys()),
                        value="VAIS1000 Medium",
                        label="Chọn giọng đọc"
                    )
                    length_scale = gr.Slider(0.1, 2.0, value=0.7, label="Tốc độ nói (Length Scale)")
                    noise_scale = gr.Slider(0.1, 1.5, value=0.8, label="Độ biểu cảm (Noise Scale)")
                    noise_w = gr.Slider(0.1, 1.5, value=0.8, label="Độ ổn định (Noise W)")
                    sentence_silence = gr.Slider(0.0, 2.0, value=0.5, label="Khoảng lặng (Sentence Silence)")
                    
                    submit_btn = gr.Button("▶️ Kể Truyện (Tạo Audio)", elem_id="submit-btn")
                    
                    gr.Markdown("---")
                    gr.Markdown("## 🔊 Kết quả")
                    # Dùng Label để giữ bố cục, nhưng nó sẽ trả về rỗng theo logic mới
                    output_message = gr.Label("", elem_id="output_text") 
                    audio_output = gr.Audio(label="Audio Đầu Ra", type="filepath")
                    
            gr.Markdown("### 📜 Lịch sử Phiên làm việc (Session History)")
            history_table = gr.Dataframe(
                headers=["Text", "Model", "Audio"], 
                datatype=["str","str","file"], 
                row_count=(1,10),
                value=initial_history_table_value
            )

        # 🟢 TAB 2: LỊCH SỬ TỔNG QUAN
        with gr.TabItem("📖 Lịch Sử Đã Tạo"):
            gr.Markdown("⚡ Lịch sử text → audio (mới nhất lên trên)")
            history_table2 = gr.Dataframe(
                headers=["Text", "Model", "Audio"], 
                datatype=["str","str","file"], 
                row_count=(1,10),
                value=initial_history_table_value
            )

    # --- Logic ---
    def show_story(story_name):
        return STORIES[story_name]

    story_dropdown.change(show_story, inputs=story_dropdown, outputs=story_preview)

    # Ẩn/hiện textbox & dropdown khi đổi chế độ
    def toggle_input_mode(mode):
        if mode == "Nhập văn bản":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    mode.change(
        toggle_input_mode, 
        inputs=mode, 
        outputs=[story_dropdown, story_preview, text_input]
    )

    # Khi nhấn "Tạo giọng nói"
    def read_text_or_story(mode, story_name, story_preview_content, custom_text, model_choice, length_scale, noise_scale, noise_w, sentence_silence):
        
        text = ""
        if mode == "Nhập văn bản":
            text = custom_text.strip()
        else:
            text = story_preview_content.strip()
        
        # Output cho output_message sẽ là chuỗi rỗng khi thành công/thất bại
        empty_msg = "" 
        
        if not text:
            # Nếu text rỗng, trả về thông báo lỗi, audio None, và table khởi tạo
            return "⚠️ Vui lòng nhập nội dung!", None, initial_history_table_value, initial_history_table_value
            
        # Hàm tts_and_save trả về 3 giá trị (audio, table1, table2)
        audio, table1, table2 = tts_and_save(text, model_choice, length_scale, noise_scale, noise_w, sentence_silence)
        
        # Nếu audio là None (thất bại), trả về thông báo lỗi chung
        if audio is None:
             error_msg = "❌ Lỗi: Không thể tạo audio. Kiểm tra console."
             return error_msg, None, table1, table2
        
        # Trả về msg rỗng khi thành công
        return empty_msg, audio, table1, table2
        
    submit_btn.click(
        read_text_or_story,
        # Cập nhật danh sách inputs để thêm story_preview
        inputs=[mode, story_dropdown, story_preview, text_input, model_dropdown, length_scale, noise_scale, noise_w, sentence_silence],
        # Outputs: msg, audio, table1, table2
        outputs=[output_message, audio_output, history_table, history_table2], 
    )


if __name__ == "__main__":
    demo.launch()
