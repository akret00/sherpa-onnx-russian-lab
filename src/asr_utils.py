"""Модуль с функциями для работы ASR"""
from abc import ABC, abstractmethod
import numpy
import sherpa_onnx
from config import SR, PipelineConfig
from entities import AudioSegment


# =====================================================================
# 1. ИНТЕРФЕЙС (Абстрактный базовый класс BaseASR)
# =====================================================================
class BaseASR(ABC):
    """
    Абстрактный интерфейс, определяющий контракт для всех ASR моделей.
    Наследование от него гарантирует проверку сигнатур во время разработки.
    """
    @abstractmethod
    def decode_asr(
        self, samples_f32: numpy.ndarray | None = None, t_start: float | None = None
    ) -> str:
        """
        Распознает сегмент аудио в текст
        Параметр samples_f32 нужен только для реального ASR
        Параметр t_start нужен только для режима Оракула
        """


# =====================================================================
# 2. АДАПТЕР ДЛЯ НАСТОЯЩЕЙ БИБЛИОТЕКИ SHERPA-ONNX
# =====================================================================
class SherpaASRAdapter(BaseASR):
    """
    Безопасная обертка над оригинальным бинарным C++ классом sherpa_onnx.
    Приводит его к нашей строгой Python-иерархии.
    """
    def __init__(self, pl_config: PipelineConfig) -> None:
        # Инициализируем настоящий бинарный объект ASR внутри
        if pl_config.asr.model_type == "nemo_ctc":
            self._real_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model = pl_config.asr.nemo_model_path,
                tokens = pl_config.asr.nemo_tokens_path,
                num_threads = pl_config.runtime.num_threads,
                sample_rate = SR,
                feature_dim = 80,
                provider = pl_config.runtime.provider,
            )
        elif pl_config.asr.model_type == "qwen3":
            self._real_recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend = pl_config.asr.qwen3_conv_frontend_path,
                encoder = pl_config.asr.qwen3_encoder_path,
                decoder = pl_config.asr.qwen3_decoder_path,
                tokenizer = pl_config.asr.qwen3_tokenizer_path,
                num_threads = pl_config.runtime.num_threads,
                sample_rate = SR,
                feature_dim = 128,
                provider = pl_config.runtime.provider,
            )
        else:
            raise ValueError(f"Unknown ASR type: {pl_config.asr.model_type}")

    def decode_asr(
        self, samples_f32: numpy.ndarray | None = None, t_start: float | None = None
    ) -> str:
        """Распознает сегмент аудио в текст"""
        stream = self._real_recognizer.create_stream()
        stream.accept_waveform(SR, samples_f32)
        self._real_recognizer.decode_stream(stream)
        return str(stream.result.text.strip())


# =====================================================================
# 3. КЛАСС ОРАКУЛ (ASR по разметке для бенчмарка диаризации и тестов)
# =====================================================================
class OracleASR(BaseASR):
    """ASR-Оракул, который ориентируется на идеальные тайминги разметки."""
    def __init__(self, padding_seconds: float = 0.0) -> None:
        """
        padding_seconds нужен для того, что бы скомпенсировать отклонения разметки из за
        квантования текущего времени аудио на размер окна
        """
        self.padding = padding_seconds
        self.markup_segments: list[AudioSegment] = []
        self.current_markup_segment_num = 0

    def set_markup_segments(self, markup_segments: list[AudioSegment]) -> None:
        """Специфичный метод Оракула для загрузки ручной разметки фраз."""
        # Список markup_segments уже должен быть отсортирован по времени начала сегмента
        # и проверен на отрицательную длительность сегментов
        self.markup_segments = markup_segments

    def decode_asr(
        self, samples_f32: numpy.ndarray | None = None, t_start: float | None = None
    ) -> str:
        """Имитация Оракула: возвращает текст всех подошедших по времени сегментов"""
        if t_start is None:
            raise ValueError("Не задано значение аргумента t_start")

        collected_texts = []

        # Итерируемся по сегментам, пока не упремся в конец списка или в будущее время
        while self.current_markup_segment_num < len(self.markup_segments):
            seg = self.markup_segments[self.current_markup_segment_num]

            # Если сегмент уже должен был прозвучать
            if t_start >= seg.start_time - self.padding:
                if seg.text:  # Защита от None или пустых строк
                    collected_texts.append(seg.text.strip())
                self.current_markup_segment_num += 1
            else:
                # Сегмент из будущего, прекращаем сбор для текущего вызова
                break

        return " ".join(collected_texts)

    def reset(self) -> None:
        """Сброс состояния Оракула перез новым циклом"""
        self.current_markup_segment_num = 0
