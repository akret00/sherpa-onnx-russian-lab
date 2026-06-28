"""Модуль с датаклассами для настроек и результатов экспериментов"""
from dataclasses import dataclass
from pathlib import Path
from entities import PipelineResult


@dataclass
class ExperimentSpec:
    """
    Атомарная спецификация для тестирования одного аудиофайла.
    Полностью описывает КТО, ЧТО и КАК обрабатывает.
    """
    spec_id: str        # Уникальный ID задачи (например, "sample_001__all_oracle")
    # Данные
    audio_path: Path
    ground_truth_path: Path | None = None       # Путь к эталонной разметке (JSON/RTTM)
    # Символические имена моделей из конфига
    asr_model_name: str | None = None           # Например: "whisper-large-v3"
    embedding_model_name: str | None = None     # Например: "pyannote-wespeaker"
    vad_model_name: str = "silero"              # По умолчанию Silero
    # Матрица включения Оракулов
    use_oracle_vad: bool = False
    use_oracle_asr: bool = False
    use_oracle_diarization: bool = False
    # Профиль настроек гиперпараметров из конфига
    profile: str | None = None              # Например: meeting или noisy_environment
    # Метаданные для аналитики метрик
    metadata: dict[str, any] | None = None  # {"dataset_name": "voxceleb", "snr_level": "low"}
    # Датасет
    dataset_version: str = "0.1"
    # Пайплайн
    # Профиль нормализации текста (наверное это на попозже)
    # Метрики (wer, cer, der)

@dataclass
class PipelineResultExperiment(PipelineResult):
    """Модель для результатов эксперимента"""
    dataset_version: str = "0.1"
