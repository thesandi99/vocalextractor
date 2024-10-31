import os
import argparse
import time  # Import the time module
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash
import torch as th
import torchaudio
from werkzeug.utils import secure_filename
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model_from_args, ModelLoadingError

# Define output directory
OUTPUT_DIR = Path("./output")
UPLOAD_FOLDER = Path("./input_audio")
ALLOWED_EXTENSIONS = {"mp3", "wav"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key_here")  # Use environment variable

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model once at startup
device = "cuda" if th.cuda.is_available() else "cpu"
args = argparse.Namespace(
    model="htdemucs",
    device=device,
    shifts=1,
    overlap=0.25,
    stem=None,
    int24=False,
    float32=False,
    clip_mode="rescale",
    mp3=False,
    mp3_bitrate=320,
    filename="{track}/{stem}.{ext}",
    split=True,
    segment=None,
    name="htdemucs",
    repo=None
)

# Load model outside of the route
try:
    model = get_model_from_args(args)
    model.eval().to(device)  # Move the model to the appropriate device
except ModelLoadingError as error:
    print(f"Model loading error: {error}")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = UPLOAD_FOLDER / filename
            input_path.parent.mkdir(exist_ok=True)

            # Save uploaded file
            file.save(str(input_path))
            flash(f"Uploaded file saved as: {input_path}")

            # Run separator function
            try:
                separator([input_path], OUTPUT_DIR)
                flash(f"Output files saved in: {OUTPUT_DIR}")
            except Exception as e:
                flash(f"Error processing file: {str(e)}")
            return redirect(url_for("upload_file"))

    return render_template("upload.html")

def load_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(str(file_path))
        return waveform.to(device), sample_rate  # Move waveform to the appropriate device
    except Exception as e:
        flash(f"Error loading audio file: {str(e)}")
        return None, None

def separator(tracks, out_dir, max_duration=170):
    out_dir.mkdir(exist_ok=True)  # Ensure output directory exists
    for track in tracks:
        wav, sample_rate = load_audio(track)
        if wav is None:  # Skip if loading failed
            continue

        # Limit to the specified max_duration
        max_samples = int(sample_rate * max_duration)
        wav = wav[:, :max_samples]

        # Start timing
        start_time = time.time()  # Record the start time

        # Process with the model
        sources = apply_model(model, wav[None], device=device, shifts=1, overlap=0.25)[0]

        # End timing
        end_time = time.time()  # Record the end time
        processing_time = end_time - start_time  # Calculate the duration

        # Print the time taken
        print(f"Time taken to process {track.name}: {processing_time:.2f} seconds")  # Print the duration

        for source, name in zip(sources, model.sources):
            output_file = out_dir / f"{track.stem}_{name}.mp3"
            save_audio(source[:, :max_samples], str(output_file), samplerate=sample_rate, bitrate=args.mp3_bitrate)
            flash(f"Separated {name} saved as {output_file}")

if __name__ == "__main__":
    app.run(debug=True)
