from flask import Flask, request, render_template, send_file
import requests
import os
import tempfile
import csv
import zipfile

app = Flask(__name__)
API_KEY = os.environ.get("NARAKEET_API_KEY")

def text_to_speech(text, voice="mickey"):
    url = f"https://api.narakeet.com/text-to-speech/m4a?voice={voice}"
    options = {
        "headers": {
            "Accept": "application/octet-stream",
            "Content-Type": "text/plain",
            "x-api-key": API_KEY,
        },
        "data": text.encode("utf-8"),
    }
    response = requests.post(url, **options)
    response.raise_for_status()
    return response.content

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "Vui lòng upload file CSV!"

        # Tạo thư mục tạm để chứa file audio
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "tts_results.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            reader = csv.DictReader(file.read().decode("utf-8").splitlines())
            for row in reader:
                text = row.get("text")
                voice = row.get("voice", "mickey")
                if not text:
                    continue  # bỏ qua dòng trống
                audio_data = text_to_speech(text, voice)

                filename = f"{row.get('id','line')}.m4a"
                file_path = os.path.join(tmp_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                zipf.write(file_path, arcname=filename)

        return send_file(zip_path, as_attachment=True, download_name="tts_results.zip")

    return render_template("index.html")

if __name__ == "__main__":
    if not API_KEY:
        raise Exception("Bạn cần set biến môi trường NARAKEET_API_KEY trước khi chạy!")
    app.run(host="0.0.0.0", port=5000, debug=True)
