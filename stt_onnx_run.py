import argparse
import os

import librosa
import onnxruntime
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "stt_onnx")
DEFAULT_AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "transcripts")


def resolve_provider(provider):
    available = onnxruntime.get_available_providers()
    if provider not in available:
        print(f"WARNING: {provider} not available, falling back to CPUExecutionProvider")
        return "CPUExecutionProvider"
    return provider


def load_model(model_dir, provider):
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        provider=provider,
        local_files_only=True,
    )
    return processor, model


def transcribe(audio_path, processor, model):
    audio_array, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated_ids = model.generate(inputs["input_features"])
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def find_audio_files(audio_dir):
    entries = sorted(os.listdir(audio_dir))
    return [
        os.path.join(audio_dir, name)
        for name in entries
        if name.lower().endswith(AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(audio_dir, name))
    ]


def transcribe_directory(
    audio_dir=DEFAULT_AUDIO_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    provider="CUDAExecutionProvider",
):
    audio_dir = os.path.abspath(audio_dir)
    model_dir = os.path.abspath(model_dir)
    output_dir = os.path.abspath(output_dir)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    processor, model = load_model(model_dir, resolve_provider(provider))

    results = {}
    for index, audio_path in enumerate(audio_files, start=1):
        name = os.path.basename(audio_path)
        print(f"[{index}/{len(audio_files)}] Transcribing {name}...")
        try:
            transcription = transcribe(audio_path, processor, model)
        except Exception as error:
            print(f"  FAILED: {error}")
            continue

        output_path = os.path.join(
            output_dir, os.path.splitext(name)[0] + ".txt"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription + "\n")

        results[name] = transcription
        print(f"  -> {output_path}")
        print(f"  {transcription}")

    print(f"\nTranscribed {len(results)}/{len(audio_files)} file(s) into {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe every audio file in a folder with the Uzbek Whisper ONNX model."
    )
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    args = parser.parse_args()

    transcribe_directory(
        audio_dir=args.audio_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        provider=args.provider,
    )
