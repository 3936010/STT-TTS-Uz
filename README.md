# STT-TTS-Uz

State-of-the-art Speech-to-Text (STT) and Text-to-Speech (TTS) pipelines tailored for the Uzbek language.

## Overview

This project provides highly optimized script runners for STT and TTS models utilizing ONNX Runtime and Hugging Face's `optimum`. The core objective is to deliver high-quality, fast, and scalable Voice AI solutions for the Uzbek language with full GPU acceleration support.

## Models

You can download the pre-trained ONNX models required for STT and TTS from the following link:
🔗 **[Download Models (Google Drive)](https://drive.google.com/drive/folders/1z9gz0eWtNS8WW5PqmII7IZWRl9mCN0Aw?usp=sharing)**

## Speech-to-Text (STT)

Our STT model has been tested for accurate transcription of the Uzbek language. The pipeline accurately converts spoken Uzbek into high-quality text.

### STT Inference Result

When running our inference script on the test audio file `audio/test.wav`, the model generates the following output:

**Result:**
> "assalomu alaykum. men navoiy ovozli modeliman. siz kiritgan matnni tabiiy va ravon ovozga aylantiraman. har xil jumlalarni o'qib, talaffuzimni sinab ko'rishingiz mumkin. masalan, bugun ob havo quyoshli va yoqimli."

## Text-to-Speech (TTS)

Our TTS system accurately synthesizes realistic and naturally flowing Uzbek speech from text. The generated audio files can be found in the `outputs/` folder.

### TTS Generated Audio

You can listen to generated output audio files directly below:

1. **Standard Model:** [test_onnx_output.wav](outputs/test_onnx_output.wav)
2. **Fine-Tuned Model:** [test_onnx_output_finetuned.wav](outputs/test_onnx_output_finetuned.wav)

*(Note: Clicking on the links above will open the GitHub media player where you can directly listen to the audio).*

## Tech Stack

- **Python 3.13**
- **ONNX Runtime (GPU):** For accelerated model inference.
- **Hugging Face Optimum:** For seamless ONNX model integration.
- **Librosa:** For audio loading, resampling, and feature extraction.
