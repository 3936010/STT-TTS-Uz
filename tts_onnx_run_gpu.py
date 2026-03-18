import torch
import numpy as np
import onnxruntime
from scipy.io.wavfile import write

import utils
import commons
from text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = np.array(text_norm, dtype=np.int64)
    return text_norm

def test_onnx_inference(text, model_path="./tts_onnx/vits_uzbek_finetuned.onnx", output_path="./outputs/test_onnx_output_finetuned.wav", provider="CUDAExecutionProvider"):
    print(f"Loading configuration...")
    hps = utils.get_hparams_from_file("./tts_onnx/uz_tts_base.json")
    
    # Check available providers
    available = onnxruntime.get_available_providers()
    print(f"Available ONNX providers: {available}")
    
    if provider not in available:
        print(f"WARNING: {provider} not available, falling back to CPUExecutionProvider")
        provider = "CPUExecutionProvider"
    
    print(f"Loading ONNX session from {model_path} with {provider}...")
    ort_session = onnxruntime.InferenceSession(model_path, providers=[provider])
    
    # Confirm which provider is actually being used
    active_provider = ort_session.get_providers()[0]
    print(f"Active provider: {active_provider}")
    
    print(f"Processing text: '{text}'...")
    stn_tst = get_text(text, hps)
    
    x = np.expand_dims(stn_tst, axis=0) # [1, T]
    x_lengths = np.array([stn_tst.shape[0]], dtype=np.int64)
    noise_scale = np.array(0.667, dtype=np.float32)
    noise_scale_w = np.array(0.8, dtype=np.float32)
    length_scale = np.array(1.1, dtype=np.float32)

    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "noise_scale": noise_scale,
        "noise_scale_w": noise_scale_w,
        "length_scale": length_scale
    }

    print("Running ONNX inference...")
    ort_outs = ort_session.run(None, inputs)
    audio = ort_outs[0][0, 0] # [B, 1, T] -> [T]
    
    print(f"Successfully generated audio. Saving to: {output_path}")
    write(output_path, hps.data.sampling_rate, audio)

if __name__ == "__main__":
    test_onnx_inference("bugun havo juda yaxshi, quyosh borlab turibdi! Men uyga ketyapman. O'g'iloy g'alati bo'lib yuribdi.")
