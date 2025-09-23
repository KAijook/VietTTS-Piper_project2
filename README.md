
## 🎶 Vietnamese TTS with Piper + Gradio  

Ứng dụng Text-to-Speech (TTS) tiếng Việt chất lượng cao, sử dụng **[Piper](https://github.com/rhasspy/piper)** và giao diện **Gradio**.  
👉 Bạn có thể nhập văn bản tiếng Việt và nhận file âm thanh `.wav` với giọng tự nhiên, có hậu xử lý (noise reduction, compressor, reverb).  

---

# 🚀 Tính năng  

- ✅ Hỗ trợ **Piper ONNX model** (`vi_VN-25hours_single-low.onnx`).  
- ✅ Giao diện **Gradio** với nhiều tuỳ chỉnh: tốc độ, độ biểu cảm, độ ổn định, khoảng lặng.  
- ✅ **Hậu xử lý âm thanh**:  
  - Noise reduction (khử nhiễu nền).  
  - Compressor (cân bằng âm lượng).  
  - Reverb nhẹ (âm tự nhiên hơn).  
- ✅ Lưu **lịch sử văn bản → audio** để tải lại khi cần.  
- ✅ Giao diện **pastel dark theme** hiện đại, header có thể nhấp nháy ✨.  

---

# 📦 Cài đặt  

Yêu cầu: Python 3.9+  

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# Cài thư viện cần thiết
pip install -r requirements.txt

# Sử dụng

1. Nhập văn bản tiếng Việt vào ô text
2. Điều chỉnh các tham số theo ý muốn:
   - **Tốc độ nói**: Giá trị thấp = nói nhanh hơn
   - **Độ biểu cảm**: Giá trị cao = biểu cảm hơn
   - **Độ ổn định**: Giá trị cao = ổn định hơn
   - **Khoảng lặng**: Thời gian nghỉ giữa các câu
3. Nhấn "Tạo giọng nói"
4. Nghe kết quả và tải về nếu cần

# Mô hình

Sử dụng mô hình `vi_VN-25hours_single-low` được huấn luyện trên 25 giờ dữ liệu tiếng Việt.

# License

Không dùng cho mục đích thương mại
