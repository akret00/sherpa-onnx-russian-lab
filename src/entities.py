""" КЛАССЫ ДАННЫХ (ОБЪЕКТНЫЕ МОДЕЛИ) """
from dataclasses import dataclass, field
from datetime import datetime
import numpy
from config import PipelineConfig

@dataclass(kw_only=True)
class SpeakerEmbedding:
    """Эмбеддинг спикера для конкретной модели."""
    id: int | None = None
    speaker_id: int | None = None  # Связь с сущностью Speaker (Внешний ключ)
    model_name: str = ""           # Короткое имя модели (например, "pyannote_v3")
    embedding: numpy.ndarray
    created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class Speaker:
    """Модель данных спикера."""
    id: int | None = None
    name: str | None = None
    embeddings: list[SpeakerEmbedding] = field(default_factory=list)
    total_count: int = 0 # Глобальный счетчик фраз
    count: int = 0  # Сессионный счетчик фраз, не сохраняется в БД
    created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_embedding(self, model_name: str) -> numpy.ndarray | None:
        """Возвращает вектор для конкретной модели или None, если его нет."""
        for emb in self.embeddings:
            if emb.model_name == model_name:
                return emb.embedding
        return None

    def add_embedding(self, model_name: str, embedding: numpy.ndarray) -> None:
        """Добавляет новый эмбеддинг или перезаписывает старый для этой модели."""
        # Сначала проверяем, нет ли уже такой модели, чтобы не плодить дубли
        for emb in self.embeddings:
            if emb.model_name == model_name: # Если модель найдена, обновляем эмбеддинг
                emb.embedding = embedding
                return

        # Если модели нет, добавляем новую сущность в список
        self.embeddings.append(
            SpeakerEmbedding(speaker_id = self.id, model_name = model_name, embedding = embedding)
        )

@dataclass
class AudioFile:
    """Модель метаданных аудиофайла."""
    id: int | None = None
    file_path: str | None = None
    duration_seconds: float = 0.0
    processed_at: str | None = None
    segments: list[AudioSegment] | None = None

@dataclass
class AudioSegment:
    """Модель текстового сегмента аудиофайла с таймкодами."""
    id: int | None = None
    audio_file_id: int | None = None
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
    pl_config : PipelineConfig | None = None
    speakers: list[Speaker] | None = None
    file: AudioFile | None = None
    segments: list[AudioSegment] | None = None
    markup_segments: list[AudioSegment] | None = None
    start_time: datetime  = datetime.now()  # Время запуска пайплайна
    proc_time: float | None = None          # Время работы пайплайна
    total_ram: float | None = None          # Объем занимаемой памяти в ОЗУ, в МБ
    sherpa_version: str | None = None       # Номер версии пакета sherpa_onnx
