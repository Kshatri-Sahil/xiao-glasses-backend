import os
import cv2
import numpy as np
import requests
import io
import wave
import base64
from flask import Flask, request, jsonify
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import face_recognition
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr

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

# --- FEATURE 2: CLOUD SPEECH & RAG RULES PIPELINE ---
def transcribe_pcm_audio(pcm_bytes):
    """
    Transcribes raw 16kHz 16-bit Mono PCM bytes with zero local RAM overhead.
    """
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_bytes)
        wav_buffer.seek(0)

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        print(f"[+] Transcribed Speech: '{text}'")
        return text
    except sr.UnknownValueError:
        print("[!] Speech Recognition: Inaudible or quiet audio")
        return ""
    except Exception as e:
        print(f"[!] Speech Recognition Error: {e}")
        return ""


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
    if not CHUNKS or VECTORIZER is None or not query:
        return ""
    try:
        query_vector = VECTORIZER.transform([query])
        similarity_scores = (TEXT_MATRIX * query_vector.T).toarray().flatten()
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        return "\n".join([CHUNKS[i] for i in top_indices])
    except Exception:
        return ""


# --- API ENDPOINTS ---
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Smart Glasses AI Server Online",
        "indexed_players": known_face_names,
        "rag_rules_loaded": len(CHUNKS) > 0
    }), 200


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
                    player_slug, {"Sport": "Pickleball", "Skill Level": "Player Profile"}
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


@app.route("/api/players", methods=["GET"])
def get_players():
    return jsonify({
        "players": list(PLAYER_DATABASE.keys()),
        "database": PLAYER_DATABASE,
        "indexed_faces": known_face_names
    }), 200


@app.route("/api/register_player", methods=["POST"])
def register_player():
    """
    Accepts multipart/form-data with 'name', 'image', and optional stats.
    Encodes the new face dynamically into RAM and saves to disk with zero server reboot!
    """
    try:
        if "image" in request.files and "name" in request.form:
            name = request.form["name"].strip().lower().replace(" ", "_")
            file = request.files["image"]
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Invalid image file format"}), 400
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            
            if not encodings:
                return jsonify({"error": "No face detected in the uploaded photo"}), 400
                
            save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
                
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            
            stats = {
                "Sport": request.form.get("sport", "Pickleball"),
                "Skill Level": request.form.get("skill_level", "Intermediate"),
                "Matches Played": request.form.get("matches", "1"),
                "Win Ratio": request.form.get("win_ratio", "50%"),
                "Preferred Shot": request.form.get("preferred_shot", "Serve")
            }
            PLAYER_DATABASE[name] = stats
            
            return jsonify({
                "status": "success",
                "message": f"Player '{name.replace('_', ' ').title()}' successfully registered in Cloud AI!",
                "stats": stats
            }), 200
        else:
            return jsonify({"error": "Missing 'name' or 'image' file in request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask_rules", methods=["POST"])
def ask_rules_offline():
    if not request.data:
        return jsonify({"error": "Missing raw audio byte stream"}), 400
    try:
        print(
            f"[*] Received {len(request.data)} bytes of streaming audio from glasses mic..."
        )
        player_question = transcribe_pcm_audio(request.data)

        if not player_question.strip():
            answer_text = "I didn't catch that clearly. Please hold the button and speak into the glasses microphone."
            player_question = "(No speech captured)"
        else:
            context = retrieve_relevant_context(player_question, top_k=2)
            context_prompt = f"\nRelevant Rulebook Excerpts:\n{context}\n" if context else ""

            prompt = (
                "You are an expert sports referee for Pickleball. "
                "Answer the user's rule query clearly and authoritatively in 1 or 2 short sentences suitable for text-to-speech audio.\n"
                f"{context_prompt}"
                f"Question: {player_question}"
            )
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            }
            
            models_to_try = [
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "mistralai/mistral-7b-instruct:free",
                "nvidia/nemotron-3-nano:free"
            ]
            answer_text = ""
            for model_name in models_to_try:
                try:
                    openrouter_response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model_name, 
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=8
                    ).json()
                    if "choices" in openrouter_response and len(openrouter_response["choices"]) > 0:
                        answer_text = openrouter_response["choices"][0]["message"]["content"].strip()
                        print(f"[+] LLM Response from {model_name}: {answer_text}")
                        break
                except Exception as llm_err:
                    print(f"[!] Error querying {model_name}: {llm_err}")
            
            if not answer_text:
                answer_text = "According to USA Pickleball rules, all volleys must be initiated outside the non-volley zone, and the ball must bounce once per side after the serve."

        output_mp3 = "response.mp3"
        output_wav = "response.wav"

        # 1. Generate natural speech using Google TTS
        tts = gTTS(text=answer_text, lang="en")
        tts.save(output_mp3)

        # 2. Amplify and convert to 16kHz 16-bit Mono WAV for ESP32 I2S compatibility
        audio = AudioSegment.from_mp3(output_mp3)
        audio = audio + 15 # Increase volume by 15 dB
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_wav, format="wav")

        with open(output_wav, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(
            f"[*] Sending AI response: Question='{player_question}' | Answer='{answer_text}'"
        )
        return jsonify({
            "status": "success",
            "transcribed_question": player_question,
            "answer_text": answer_text,
            "audio_base64": audio_b64
        }), 200
    except Exception as e:
        print(f"[!] ask_rules exception: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    data = request.json or {}
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
            
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return jsonify({
            "status": "success",
            "text": text,
            "audio_base64": audio_b64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Unified Smart Glasses Server Pipeline on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)