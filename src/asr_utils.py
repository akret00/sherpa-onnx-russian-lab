"""Модуль с функциями для работы ASR"""
import numpy as np
import sherpa_onnx
import config

def decode_asr(recognizer: sherpa_onnx.OfflineRecognizer, samples_f32: np.ndarray) -> str:
    """Распознает сегмент аудио в текст"""
    stream = recognizer.create_stream()
    stream.accept_waveform(config.SR, samples_f32)
    recognizer.decode_stream(stream)
    return str(stream.result.text.strip())
