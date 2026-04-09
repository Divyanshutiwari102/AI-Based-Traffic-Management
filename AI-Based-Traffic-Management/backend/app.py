from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import CORS_ORIGIN, MAX_UPLOAD_VIDEOS, UPLOAD_DIR
from tasks import get_task_status, process_videos_task

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGIN}})


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('videos')
    if len(files) != MAX_UPLOAD_VIDEOS:
        return jsonify({'error': f'Exactly {MAX_UPLOAD_VIDEOS} videos required'}), 400

    video_paths = []
    for i, file in enumerate(files):
        filename = secure_filename(file.filename or f'video_{i}.mp4')
        output_path = Path(UPLOAD_DIR) / f'{i}_{filename}'
        file.save(str(output_path))
        video_paths.append(str(output_path))

    job = process_videos_task.delay(video_paths)
    return jsonify({'job_id': job.id}), 202


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id: str):
    status = get_task_status(job_id)
    return jsonify(status), (404 if status.get("state") == "NOT_FOUND" else 200)


if __name__ == '__main__':
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True)
