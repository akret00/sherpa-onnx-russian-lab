"""Модуль с функциями для работы VAD"""
import numpy as np
from config import SR

def get_speec_segments(vad):
    """Извлекает сегменты из VAD и возвращает их как генератор."""
    # Пока VAD не пустой, то есть содержит законченные фразы
    while not vad.empty():
        # Внимание: сохраните данные перед vad.pop()
        # Извлекаем данные для очередной законченной фразы
        start_sample = vad.front.start
        samples = np.array(vad.front.samples, dtype=np.float32)
        vad.pop()

        # Вычисляем тайминги
        t_start = start_sample / SR
        t_end = (start_sample + len(samples)) / SR

        yield samples, t_start, t_end
