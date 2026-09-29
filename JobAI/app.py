import os
from flask import Flask, request, jsonify

from config import Config
from database.database import init_db
from database.candidate import save_candidate_profile
from ai.resume_parser import parse_resume
from ai.candidate_profile import build_candidate_profile

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return jsonify({"status": "JobAI is running", "phase": 1})


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    """
    Accepts a PDF resume, runs the Phase 1 pipeline
    (parse -> candidate profile -> save to DB), and returns the result.
    """
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as form field 'resume'."}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF resumes are supported right now."}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    resume_data = parse_resume(save_path)
    profile = build_candidate_profile(resume_data)
    candidate_id = save_candidate_profile(profile, resume_path=save_path)

    return jsonify({"candidate_id": candidate_id, "profile": profile.to_dict()}), 201


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
