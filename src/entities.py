""" КЛАССЫ ДАННЫХ (ОБЪЕКТНЫЕ МОДЕЛИ) """
from dataclasses import dataclass
import numpy

@dataclass
class Speaker:
    """Модель данных спикера."""
    id: int | None = None
    name: str = "Unknown Speaker"
    embedding: numpy.ndarray | None = None
    total_count: int = 0 # Глобальный счетчик фраз
    count: int = 0  # Сессионный счетчик фраз, не сохраняется в БД
    created_at: str | None = None

@dataclass
class AudioFile:
    """Модель метаданных аудиофайла."""
    id: int | None = None
    file_path: str = ""
    duration_seconds: float = 0.0
    processed_at: str | None = None
    segments: list[AudioSegment] | None = None

@dataclass
class AudioSegment:
    """Модель текстового сегмента аудиофайла с таймкодами."""
    id: int | None = None
    audio_file_id: int | None = None
    audio_file: AudioFile | None = None
    speaker_id: int | None = None
    speaker: Speaker | None = None
    cos_similarity: float = 0.0 # Косинусная схожесть между эмбеддингами спикера и сегмента
    start_time: float = 0.0
    end_time: float = 0.0
    text: str | None = None
    word_count: int | None = None

@dataclass
class PipelineResult:
    """Модель с результатами работы пайплайна"""
    pipeline_type: str | None
    speakers: list[Speaker]
    file: AudioFile | None
    segments: list[AudioSegment] | None
    run_time: float = 0
    markup_segments: list[AudioSegment] | None = None
