"""Модуль с функциями для работы ASR"""
import numpy as np
import sherpa_onnx
import config

def compute_embedding(extractor: sherpa_onnx.SpeakerEmbeddingExtractor, samples_f32: np.ndarray) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=config.SR, waveform=samples_f32)
    stream.input_finished()
    # extractor.is_ready(stream) обычно True, если сегмент не слишком короткий
    emb = extractor.compute(stream)
    return np.array(emb, dtype=np.float32)

def decode_asr(recognizer: sherpa_onnx.OfflineRecognizer, samples_f32: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(config.SR, samples_f32)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()
