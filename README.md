# STT-TTS-Uz

Speech-to-Text (STT) and Text-to-Speech (TTS) pipelines for the Uzbek language.

## Overview

This project provides Python scripts to run STT and TTS models utilizing ONNX Runtime and Hugging Face's `optimum`. The scripts support GPU acceleration for faster inference.

## Models

You can download the pre-trained ONNX models required for STT and TTS from the following link:
🔗 **[Download Models (Google Drive)](https://drive.google.com/drive/folders/1z9gz0eWtNS8WW5PqmII7IZWRl9mCN0Aw?usp=sharing)**

## Usage

Here is a brief guide on how to run the models on your machine:

1. **Install Dependencies:**
   Ensure you have installed the requirements using `uv` or `pip`:
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

2. **Download Models:**
   Download the TTS and STT models from the Google Drive link provided above and extract them into your workspace.

3. **Update Paths in the Scripts:**
   Before running the inference scripts, open them and update the `model_dir` path to point to the location where you saved the downloaded models:
   - For STT: `stt_onnx_run.py`
   - For TTS: `tts_onnx_run.py` (or `tts_onnx_run_gpu.py`)

4. **Run the Inference Scripts:**
   Once the paths are set, you can run the files via the command line:

   ```bash
   python stt_onnx_run.py
   python tts_onnx_run.py
   ```

## Speech-to-Text (STT)

### STT Inference Result

When running our inference script on the test audio file `audio/test.wav`, the model generates the following output:

**Result:**
> "assalomu alaykum. men navoiy ovozli modeliman. siz kiritgan matnni tabiiy va ravon ovozga aylantiraman. har xil jumlalarni o'qib, talaffuzimni sinab ko'rishingiz mumkin. masalan, bugun ob havo quyoshli va yoqimli."

## Text-to-Speech (TTS)

The generated audio outputs can be found in the `outputs/` folder.

### TTS Generated Audio

You can listen to generated output audio files directly below:

1. **Standard Model:** [test_onnx_output.wav](outputs/test_onnx_output.wav)
2. **Fine-Tuned Model:** [test_onnx_output_finetuned.wav](outputs/test_onnx_output_finetuned.wav)

*(Note: Clicking on the links above will open the GitHub media player where you can directly listen to the audio).*

## Tech Stack

- **Python 3.13**
- **ONNX Runtime (GPU):** For model inference.
- **Hugging Face Optimum:** For ONNX model integration.
- **Librosa:** For audio loading, resampling, and feature extraction.
