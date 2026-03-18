import os
import librosa
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

def test_stt_inference(
    audio_path,
    model_dir="/home/sd/STT-TTS-Uz/stt_onnx",
    provider="CUDAExecutionProvider",
):
    model_dir = os.path.abspath(model_dir)

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        provider=provider,
        local_files_only=True,
    )

    print(f"Processing: {audio_path}")
    audio_array, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating transcription...")
    generated_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("--- Result ---")
    print(transcription)
    return transcription

if __name__ == "__main__":
    test_stt_inference("/home/sd/STT-TTS-Uz/audio/test.wav")