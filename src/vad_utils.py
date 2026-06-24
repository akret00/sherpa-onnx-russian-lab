"""
Модуль содержит:
- функцию извлечения сегментов из VAD, которая возвращает сегменты как генератор
- абстрактный класс BaseVAD
- OracleVAD(BaseVAD) - делит аудиопоток на основе заранее известной разметки
- SherpaVADAdapter(BaseVAD) - обертка над классом sherpa_onnx.VoiceActivityDetector
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy
from sherpa_onnx import VadModelConfig, VoiceActivityDetector
from config import SR
from entities import AudioSegment


def get_speec_segments(vad: BaseVAD):
    """Извлекает сегменты из VAD и возвращает их как генератор."""
    # Пока VAD не пустой, то есть содержит законченные фразы
    while not vad.empty():
        # Внимание: сохраните данные перед vad.pop()
        # Извлекаем данные для очередной законченной фразы
        start_sample = vad.front.start
        samples = numpy.array(vad.front.samples, dtype=numpy.float32)
        vad.pop()

        # Вычисляем тайминги
        t_start = start_sample / SR
        t_end = (start_sample + len(samples)) / SR

        yield samples, t_start, t_end


# =====================================================================
# 1. ИНТЕРФЕЙС (Абстрактный базовый класс BaseVAD)
# =====================================================================
class BaseVAD(ABC):
    """
    Абстрактный интерфейс, определяющий контракт для всех VAD-детекторов.
    Наследование от него гарантирует проверку сигнатур во время разработки.
    """
    @property
    @abstractmethod
    def front(self):
        """Возвращает свойство front"""

    @abstractmethod
    def accept_waveform(self, samples: numpy.ndarray[numpy.float32]) -> None:
        """
        Принимает новую порцию аудио-сэмплов.
        samples: np.ndarray с dtype=np.float32 и значениями в диапазоне [-1.0, 1.0]
        """

    @abstractmethod
    def empty(self) -> bool:
        """Проверить, пуст ли буфер готовых речевых сегментов."""

    @abstractmethod
    def pop(self) -> None:
        """Извлечь накопленный речевой сегмент с dtype=np.float32 для отправки в ASR."""

    @abstractmethod
    def reset(self) -> None:
        """Сбросить внутреннее состояние детектора."""


# =====================================================================
# 2. АДАПТЕР ДЛЯ НАСТОЯЩЕЙ БИБЛИОТЕКИ SHERPA-ONNX
# =====================================================================
class SherpaVADAdapter(BaseVAD):
    """
    Безопасная обертка над оригинальным бинарным C++ классом sherpa_onnx.
    Приводит его к нашей строгой Python-иерархии.
    """
    def __init__(self, config: VadModelConfig, buffer_size_in_seconds: int):
        # Инициализируем настоящий бинарный объект внутри
        # self._real_vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds)
        self._real_vad = VoiceActivityDetector(config, buffer_size_in_seconds)

    @property
    def front(self):
        """Чтение атрибута из оригинального класса"""
        return self._real_vad.front

    def accept_waveform(self, samples: numpy.ndarray) -> None:
        # Просто пробрасываем вызов в C++ (в оригинальный класс VoiceActivityDetector)
        self._real_vad.accept_waveform(samples)

    def empty(self) -> bool:
        return self._real_vad.empty()

    def pop(self) -> None:
        self._real_vad.pop()

    def reset(self) -> None:
        self._real_vad.reset()


# =====================================================================
# 3.1 СТРУКТУРА ДЛЯ ИМИТАЦИИ REFRENCES ИЗ C++
# =====================================================================
@dataclass
class FakeSpeechSegment:
    """Полная имитация оригинального C++ класса SpeechSegment из sherpa-onnx."""
    start: int = 0
    # Используем numpy array, так как np.array() от него отработает мгновенно и без ошибок
    samples: numpy.ndarray | None = None

# =====================================================================
# 3.2 КЛАСС ОРАКУЛ (VAD по разметке для бенчмарка ASR и тестов)
# =====================================================================
class OracleVAD(BaseVAD):
    """
    VAD-Оракул, который ориентируется на идеальные тайминги разметки.
    """
    def __init__(self, buffer_size_in_seconds: int = 100, padding_seconds: float = 0.0):
        self.sample_rate = SR
        self.padding = padding_seconds
        self.markup_segments: list[AudioSegment] | None = None
        self.last_sample_num = 0
        self.current_markup_segment_num : int = 0
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.speech_buffer: numpy.ndarray[numpy.float32] = numpy.zeros(
            buffer_size_in_seconds * SR,
            dtype=numpy.float32
        )

        # Логические триггеры для управления состоянием фразы
        self.in_phrase = False
        self.phrase_start_sample = 0
        # True только когда текущая фраза полностью дочитана до конца
        self.phrase_ready = False

        # Создаем публичный атрибут front, как у оригинального VAD
        self._front = FakeSpeechSegment()

    @property
    def front(self):
        """Чтение атрибута из оригинального класса"""
        return self._front

    def set_markup_segments(self, markup_segments: list[AudioSegment]) -> None:
        """Специфичный метод Оракула для загрузки ручной разметки фраз."""
        # Проверка на слишком длинные фразы
        max_segment_duration = self.buffer_size_in_seconds
        for _, seg in enumerate(markup_segments):
            duration = seg.end_time - seg.start_time
            if duration > max_segment_duration:
                raise ValueError(
                    f"Сегмент {seg.id}: {seg.start_time:.2f}-{seg.end_time:.2f}с "
                    f"длиной {duration:.2f}с превышает буфер {max_segment_duration}с"
                )

        # Список markup_segments уже должен быть отсортирован по времени начала сегмента
        # и проверен на отрицательную длительность сегментов
        self.markup_segments = markup_segments

        self.reset() # Сброс количестка накопленных сэмплов меток в ноль

    def accept_waveform(self, samples: numpy.ndarray[numpy.float32]) -> None:
        # Считаем временные границы текущего чанка
        num_samples = len(samples)
        start_chunk_time = self.last_sample_num / self.sample_rate
        self.last_sample_num += num_samples
        end_chunk_time = self.last_sample_num / self.sample_rate
        mid_chunk_time = (start_chunk_time + end_chunk_time) / 2

        # Если предыдущая фраза уже готова, но пайплайн её ещё не забрал через pop(),
        # Оракул пропускает чанки (но счетчик сэмплов крутит), до вызова pop()
        if self.phrase_ready:
            return

        # Проверяем не закончилась ли разметка, если закончилась, то выходим
        if self.current_markup_segment_num < len(self.markup_segments):
            seg = self.markup_segments[self.current_markup_segment_num]
        else:
            return

        # Проверяем, входит ли чанк в разметку (есть состояние речи или нет)
        is_speech = False
        if (seg.start_time - self.padding) <= mid_chunk_time <= (seg.end_time + self.padding):
            is_speech = True

        if not self.in_phrase:
            # Мы не во фразе, проверяем, не началась ли фраза
            if is_speech:
                # Если речь началась, то начинаем и фразу
                self.in_phrase = True
                # Фиксируем номер начального сэмпла фразы
                self.phrase_start_sample = self.last_sample_num - num_samples

        if self.in_phrase:
            write_pointer = self.last_sample_num - num_samples - self.phrase_start_sample
            # Мы во фразе, проверяем не закончилась ли фраза
            if is_speech:
                # Фраза продолжается
                # Добавляет чанк в буфер аудио
                # write_pointer = self.last_sample_num - num_samples - self.phrase_start_sample
                if write_pointer + num_samples > len(self.speech_buffer):
                    # new_capacity = len(self.speech_buffer)* 2
                    # self.speech_buffer.resize(new_capacity, refcheck=False)
                    raise ValueError(
                        f"Длина фразы в разметке превышает максимально допустимую: "
                        f"{len(self.speech_buffer) / SR}"
                    )
                self.speech_buffer[write_pointer:write_pointer + num_samples] = samples
            else:
                # Фраза закончилась
                self.in_phrase = False
                # Заполняем объект front, из которого получают результат
                self._front.start = int(self.phrase_start_sample)
                self._front.samples = self.speech_buffer[:write_pointer].copy()
                self.phrase_ready = True # Фраза готова
                # Определяем следующий ожидаемый сегмент разметки
                while True:
                    self.current_markup_segment_num += 1
                    # Если записи раметки кончились, то выходим
                    if self.current_markup_segment_num >= len(self.markup_segments):
                        return
                    # Проверяем, что время окончания сегмента разметки больше чем время
                    # текущего чанка. Иначе переходим к следующей записи разметки
                    seg = self.markup_segments[self.current_markup_segment_num]
                    if seg.end_time > mid_chunk_time:
                        break

    def empty(self) -> bool:
        """
        Возвращает False ТОЛЬКО тогда, когда фраза ПОЛНОСТЬЮ завершена 
        и готова к выдаче. В процессе накопления возвращает True.
        """
        return not self.phrase_ready

    def pop(self) -> None:
        """
        Очищает OracleVad от готовой фразы
        """
        # Полностью сбрасываем состояние готовности текущего сегмента
        self.phrase_ready = False
        self.in_phrase = False
        self.phrase_start_sample = -1
        self._front = FakeSpeechSegment()

    def reset(self) -> None:
        self.phrase_ready = False  # Сброс флага готовности
        self.in_phrase = False
        self.phrase_start_sample = 0
        self.last_sample_num = 0
        self.current_markup_segment_num = 0
        # self.speech_buffer.fill(0.0)
        self._front = FakeSpeechSegment()
