import argparse
import os

import librosa
import soundfile as sf

SOURCE_EXTENSIONS = (".ogg", ".mp3", ".m4a", ".flac")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "audio")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "audio")

# Whisper resamples everything to 16 kHz mono anyway, so convert once here.
DEFAULT_SAMPLE_RATE = 16000


def find_source_files(input_dir):
    entries = sorted(os.listdir(input_dir))
    return [
        os.path.join(input_dir, name)
        for name in entries
        if name.lower().endswith(SOURCE_EXTENSIONS)
        and os.path.isfile(os.path.join(input_dir, name))
    ]


def convert_file(source_path, output_path, sample_rate, mono):
    audio, sr = librosa.load(source_path, sr=sample_rate, mono=mono)
    if not mono and audio.ndim > 1:
        audio = audio.T  # soundfile expects [frames, channels]
    sf.write(output_path, audio, sr, subtype="PCM_16")
    return sr


def convert_directory(
    input_dir=DEFAULT_INPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    sample_rate=DEFAULT_SAMPLE_RATE,
    mono=True,
    overwrite=False,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    source_files = find_source_files(input_dir)
    if not source_files:
        print(f"No convertible audio found in {input_dir}")
        print(f"Looking for: {', '.join(SOURCE_EXTENSIONS)}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    converted = []
    for index, source_path in enumerate(source_files, start=1):
        name = os.path.basename(source_path)
        output_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".wav")

        if os.path.exists(output_path) and not overwrite:
            print(f"[{index}/{len(source_files)}] Skipping {name} "
                  f"({os.path.basename(output_path)} already exists, use --overwrite)")
            continue

        print(f"[{index}/{len(source_files)}] Converting {name}...")
        try:
            sr = convert_file(source_path, output_path, sample_rate, mono)
        except Exception as error:
            print(f"  FAILED: {error}")
            continue

        converted.append(output_path)
        channels = "mono" if mono else "stereo"
        print(f"  -> {output_path} ({sr} Hz, {channels}, 16-bit PCM)")

    print(f"\nConverted {len(converted)}/{len(source_files)} file(s) into {output_dir}")
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .ogg/.mp3 (and .m4a/.flac) files in a folder to 16-bit PCM .wav."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--stereo", action="store_true", help="Keep original channels (default: downmix to mono)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .wav files")
    args = parser.parse_args()

    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        mono=not args.stereo,
        overwrite=args.overwrite,
    )
