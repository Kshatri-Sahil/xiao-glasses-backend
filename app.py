import os
import cv2
import numpy as np
import requests
import io
import wave
from flask import Flask, request, jsonify
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import face_recognition
from gtts import gTTS
from pydub import AudioSegment

app = Flask(__name__)

KNOWN_FACES_DIR = "known_players"
pdf_path = "USAP-Official-Rulebook.pdf"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# --- FEATURE 1: FACE RECOGNITION DATABASE ---
known_face_encodings = []
known_face_names = []

print("[*] Encoding database profiles using face_recognition engine...")
for file in os.listdir(KNOWN_FACES_DIR):
    if file.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0].lower())
                print(f"[+] Successfully indexed profile: {file}")
        except Exception as face_err:
            print(f"[!] Error processing face in {file}: {face_err}")

PLAYER_DATABASE = {
    "sahil": {
        "Sport": "Pickleball",
        "DUPR Rating": "4.5 Natively",
        "Matches Played": "64",
        "Win Ratio": "78%",
        "Preferred Shot": "Third Shot Drop",
    },
    "jagmeet": {
        "Sport": "Pickleball",
        "Skill Level": "Advanced",
        "Matches Played": "39",
        "Win Ratio": "68%",
        "Preferred Shot": "Bandeja",
    },
}

# --- FEATURE 2: OFFLINE RAG TEXT PROCESSING ---
print("[*] Loading local Whisper Audio Engine on CPU...")
from faster_whisper import WhisperModel

asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")

CHUNKS = []
VECTORIZER = None
TEXT_MATRIX = None


def index_local_pdf():
    global CHUNKS, VECTORIZER, TEXT_MATRIX
    if not os.path.exists(pdf_path):
        print(
            f"[!] Target text source '{pdf_path}' missing. Rules engine will run without context."
        )
        return
    print("[*] Parsing and chunking rulebook PDF paragraphs natively...")
    try:
        reader = PdfReader(pdf_path)
        raw_text = "".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )
        lines = raw_text.split("\n")
        CHUNKS = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(current_chunk) + len(line) < 600:
                current_chunk += " " + line
            else:
                if current_chunk:
                    CHUNKS.append(current_chunk.strip())
                current_chunk = line
        if current_chunk:
            CHUNKS.append(current_chunk.strip())

        VECTORIZER = TfidfVectorizer(stop_words="english")
        TEXT_MATRIX = VECTORIZER.fit_transform(CHUNKS)
        print(
            f"[+] Local index complete. Generated {len(CHUNKS)} quick-search text segments."
        )
    except Exception as e:
        print(f"[!] Failed to parse PDF file: {e}")


index_local_pdf()


def retrieve_relevant_context(query, top_k=2):
    if not CHUNKS or VECTORIZER is None:
        return ""
    query_vector = VECTORIZER.transform([query])
    similarity_scores = (TEXT_MATRIX * query_vector.T).toarray().flatten()
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    return "\n".join([CHUNKS[i] for i in top_indices])


# --- API ENDPOINTS ---
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Smart Glasses AI Server Online"}), 200


@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    if not request.data:
        return jsonify({"face_found": False}), 400
    try:
        nparr = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"face_found": False, "error": "Invalid frame bytes"}), 400
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            if not known_face_encodings:
                continue
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6
            )
            if True in matches:
                first_match_index = matches.index(True)
                player_slug = known_face_names[first_match_index]
                name = player_slug.replace("_", " ").title()
                stats = PLAYER_DATABASE.get(
                    player_slug, {"Sport": "Racket Sports", "Status": "Guest Profile"}
                )
                return jsonify(
                    {
                        "face_found": True,
                        "name": name,
                        "box": {
                            "left": left,
                            "top": top,
                            "width": (right - left),
                            "height": (bottom - top),
                        },
                        "stats": stats,
                    }
                )
        return jsonify({"face_found": False})
    except Exception as e:
        return jsonify({"face_found": False, "error": str(e)}), 500


@app.route("/api/ask_rules", methods=["POST"])
def ask_rules_offline():
    if not request.data:
        return jsonify({"error": "Missing raw audio byte stream"}), 400
    try:
        print(
            f"[*] Received {len(request.data)} bytes of streaming audio from glasses mic..."
        )
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(request.data)
        wav_buffer.seek(0)

        segments, _ = asr_model.transcribe(wav_buffer, beam_size=1)
        player_question = "".join([seg.text for seg in segments])
        print(f"[+] Transcribed Question: '{player_question}'")

        if not player_question.strip():
            return jsonify({"error": "No clear speech captured"}), 400

        prompt = (
            "You are an expert on the game of pickleball. "
            "Please answer the following question clearly and concisely in under 2 sentences. "
            "If the question is unrelated to pickleball, try to relate it to the sport if possible, or just answer normally.\n\n"
            f"Question: {player_question}"
        )
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }
        
        openrouter_response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free", 
                "messages": [{"role": "user", "content": prompt}]
            },
        ).json()
        
        try:
            answer_text = openrouter_response["choices"][0]["message"]["content"]
        except KeyError:
            print(f"[!] OpenRouter API Error: {openrouter_response}")
            answer_text = "Sorry, I couldn't process that request."
            
        print(f"[+] Remote LLM Referee Response: {answer_text}")

        output_mp3 = "response.mp3"
        output_wav = "response.wav"

        # 1. Generate natural speech using Google TTS
        tts = gTTS(text=answer_text, lang="en")
        tts.save(output_mp3)

        # 2. Amplify and convert to 16kHz 16-bit Mono WAV for ESP32 I2S compatibility
        audio = AudioSegment.from_mp3(output_mp3)
        audio = audio + 15 # Increase volume by 15 dB
        
        # Save amplified mp3
        audio.export(output_mp3, format="mp3")
        
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_wav, format="wav")

        with open(output_wav, "rb") as f:
            audio_bytes = f.read()

        print(
            f"[*] Sending {len(audio_bytes)} bytes of WAV audio back to Flutter client!"
        )
        return audio_bytes, 200, {"Content-Type": "audio/wav"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text", "System message")
    try:
        output_mp3 = "tts_sys.mp3"
        output_wav = "tts_sys.wav"
        
        tts = gTTS(text=text, lang="en")
        tts.save(output_mp3)
        
        audio = AudioSegment.from_mp3(output_mp3)
        audio = audio + 15 # Increase volume
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_wav, format="wav")
        
        with open(output_wav, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes, 200, {"Content-Type": "audio/wav"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Unified Smart Glasses Server Pipeline on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)