"""Модуль с датаклассами для разметки, рецептов и сценариев"""
from dataclasses import dataclass
from entities import AudioSegment, AudioFile

@dataclass
class AudioSegmentMarkup(AudioSegment):
    """Модель аудиосегмента для разметки"""
    phrase_id: int | None = None

@dataclass
class AudioFileMarkup(AudioFile):
    """Модель айдиофайла для разметки"""
    dataset_version: str = "0.1"

@dataclass
class RecipeStep:
    """Модель шага рецепта"""
    step_id: int
    dictor_id: int
    phrase_id: int

@dataclass
class Recipe:
    """Модель рецепта"""
    recipe_id: str
    description: str
    pause_ms: int
    timeline: list[RecipeStep]

@dataclass
class NoiseConfig:
    """Модель описания конфигурации шума для сценария"""
    enabled: bool = False
    file: str | None = None
    snr_db: int | None = None

@dataclass
class ScenarioEvent:
    """Модель события (шага) сценария"""
    file_id: int        # В контексте генератора это будет speaker_id (или индекс + 1)
    segment_id: int     # ID исходного сегмента (AudioSegmentMarkup.id)
    start: float        # Рассчитанное время начала сегмента в сценарии (секунды)
    end: float          # Рассчитанное время конца сегмента в сценарии (секунды)
    text: str           # Текст сегмента
    gain_db: float = 0.0 # Усиление сигнала при ссборке аудио, в дБ. 0 - без изменений

@dataclass
class ScenarioEpisode:
    """Модель для описания эпизода сценария"""
    id: str
    description: str
    events: list[ScenarioEvent]
    noise: NoiseConfig | None = None

@dataclass
class Scenario:
    """Модель для описания сценария"""
    dataset_version: str = "0.1"
    sample_rate: int = 16000
    episodes: list[ScenarioEpisode] | None = None
