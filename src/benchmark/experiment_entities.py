"""Модуль с датаклассами для настроек и результатов экспериментов"""
from dataclasses import dataclass
from typing import Any
from entities import PipelineResult
from config import PipelineType


@dataclass
class ExperimentSpec:
    """
    Атомарная спецификация для тестирования одного аудиофайла.
    Полностью описывает КТО, ЧТО и КАК обрабатывает.
    """
    spec_id: str        # Уникальный ID задачи (например, "sample_001__all_oracle")
    # Данные
    audio_path: str | None = None
    ground_truth_path: str | None = None       # Путь к эталонной разметке (JSON/RTTM)
    # Символические имена моделей из конфига
    asr_model_name: str | None = None           # Например: "whisper-large-v3"
    embed_model_name: str | None = None         # Например: "pyannote-wespeaker"
    vad_model_name: str = "silero"              # По умолчанию Silero
    segmentation_model_name: str | None = None  # Например: "pyannote-int8"
    # Матрица включения Оракулов
    use_oracle_vad: bool = False
    use_oracle_asr: bool = False
    use_oracle_diarization: bool = False
    # Профиль настроек гиперпараметров из конфига
    profile: str | None = None              # Например: meeting или noisy_environment
    # Метаданные для аналитики метрик
    metadata: dict[str, Any] | None = None  # {"dataset_name": "voxceleb", "snr_level": "low"}
    # Датасет
    dataset_version: str = "0.1"
    dataset_view: str = "unknown"
    # Пайплайн
    pipeline_type: PipelineType | None = PipelineType.ASR_PIPELINE
    # Профиль нормализации текста (наверное это на попозже)
    # Метрики (wer, cer, der)
    use_wer: bool = False
    use_cer: bool = False
    use_der:bool = False
    # Способ обрезки сегментов VAD: способ обрезки (фикс или адаптивный), если фикс, то сколько

@dataclass
class PipelineResultExperiment(PipelineResult):
    """Модель для результатов эксперимента"""
    exp_spec: ExperimentSpec | None = None
    exp_id: str | None = None


@dataclass
class MetricWER:
    """Содержит метрику WER"""
    obj_id: str | None = None   # ИД метрики внутри эксперимента, может быть ИД или номер сегмента
    wer: float = 0.0
    gt_words_count: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    alignment: str | None = None

@dataclass
class MetricCER:
    """Содержит метрику CER"""
    obj_id: str | None = None   # ИД метрики внутри эксперимента, может быть ИД или номер сегмента
    cer: float = 0.0
    gt_chars_count: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    alignment: str | None = None

@dataclass
class MetricExpWER:
    """Содержит метрики WER и CER эксперимента"""
    obj_id: str | None = None   # ИД метрики внутри эксперимента, может быть ИД или номер сегмента
    seg_count: int = 0          # Количество сегментов (фраз)
    exp_wer: MetricWER | None = None    # WER для всего эксперимента
    exp_cer: MetricCER | None = None    # CER для всего эксперимента
    err_segments_wer: list[MetricWER] | None = None # Содержит метрики для сегментов с ошибками
    err_segments_cer: list[MetricCER] | None = None # Содержит метрики для сегментов с ошибками

@dataclass
class MetricDER:
    """Содержит метрику DER"""
    obj_id: str = ""         # ИД метрики внутри эксперимента, может быть ИД или номер сегмента
    der: float = 0.0
