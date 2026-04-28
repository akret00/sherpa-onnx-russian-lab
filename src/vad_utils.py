"""Модуль с функциями для работы VAD"""
import numpy as np
import asr_utils
from config import SR

def process_vad_segments(vad, recognizer):
    """
    Извлекает все завершённые фразы из VAD, распознаёт их и выводит текст в stdout.
    Args:
        vad: экземпляр VAD с накопленными сегментами
        recognizer: экземпляр ASR-распознавателя
        sample_rate: частота дискретизации аудио (Гц)
    """
    # Пока VAD не пустой, то есть содержит законченные фразы
    while not vad.empty():
        # IMPORTANT: grab data BEFORE vad.pop()
        # Извлекаем данные для очередной законченной фразы
        start_sample = vad.front.start
        seg = np.array(vad.front.samples, dtype=np.float32)
        vad.pop()

        # Распознаем (ASR) полученный из VAD сегмент
        text = asr_utils.decode_asr(recognizer, seg)

        t0 = start_sample / SR
        t1 = (start_sample + len(seg)) / SR
        if text:
            print(f"[{t0:10.3f}-{t1:10.3f}]: {text}")
