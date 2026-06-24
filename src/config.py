"""
Модуль загрузки конфигурации
"""
from dataclasses import dataclass
from pathlib import Path
import os
import yaml

# Частота дискретизации
SR = 16000
# Сбрасывать спикера, если пауза > 2 сек
MAX_PAUSE_FOR_INERTIA = 2.0
# 0.5 сек - минимальная длина сегмента с фразой, с которой начинается поиск спикера
MIN_SEARCH_SEG_LEN = 0.5
PYANNOTE_MIN_DURATION_ON = 0.3
PYANNOTE_MIN_DURATION_OFF = 0.5

# Определяем корень проекта относительно этого файла
# .parent — это папка src/, второй .parent — корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Определяем пути к ресурсам
CONFIG_PATH = BASE_DIR / "config.yaml"
BIN_DIR = BASE_DIR / "bin" # Путь для portable варианта ffmpeg
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"

if BIN_DIR.is_dir() and str(BIN_DIR) not in os.environ["PATH"]:
    # Добавляем в начало PATH
    os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ["PATH"]

# Определяем пути к файлу с базой данных и файлу со структурой базы данных
DB_DEFAULT_PATH = BASE_DIR / "db" / "speaker.sqlite3"
DB_DEFAULT_SCHEME_PATH = BASE_DIR / "src" / "db_scheme.sql"

class Config:
    """Класс загрузчика конфигурации"""
    def __init__(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        # Определяем рабочие модели
        self._vad_model_name = self._data['work']['vad_model_name']
        self._embed_model_name = self._data['work']['embed_model_name']
        self._asr_model_name = self._data['work']['asr_model_name']
        self._segmentation_model_name = self._data['work']['segmentation_model_name']

    # @property
    def get_work(self):
        """Возвращает настройки рабочих моделей"""
        return self._data.get("work", {})

    # @property
    def get_models(self):
        """Возвращает список моделей и их настроек"""
        return self._data.get("models", {})

    # @property
    def get_profiles(self):
        """Возвращает список профилей и их настроек"""
        return self._data.get("profiles", {})

    def get_vad_model(self):
        """Возвращает настройки рабочей модели VAD"""
        return self._data["models"]["vad"][self._vad_model_name]

    def get_embedding_model(self):
        """Возвращает настройки рабочей модели embedding"""
        return self._data["models"]["embedding"][self._embed_model_name]

    def get_asr_model(self):
        """Возвращает настройки рабочей модели ASR"""
        return self._data["models"]["asr"][self._asr_model_name]

    def get_segmentation_model(self):
        """Возвращает настройки рабочей модели сегментации"""
        return self._data["models"]["segmentation"][self._segmentation_model_name]

    def get_default_args(self):
        """Возвращает список параметров и их значений по умолчанию"""
        default_args = {}

        # VAD параметры
        default_args["vad-threshold"] = 0.3
        default_args["vad-min-silence"] = 0.25
        default_args["vad-min-speech"] = 0.25
        default_args["vad-max-speech"] = 10.0

        # Embedding параметры для speaker ID
        default_args["spk-threshold"] = 0.4
        default_args["min-seg-sec"] = 0.5

        return default_args

@dataclass
class PipelineConfig:
    """Конфигурация пайплайна"""
    config: Config
    num_threads = 1
    provider = "cpu"
    vad_threshold: float = 0.4 # 0.4 Было 0.3
    vad_min_silence: float = 0.1 # 0.1 Было 0.25
    vad_min_speech: float = 0.2 # 0.2 Было 0.25
    vad_max_speech: float = 10.0
    # Embedding параметры для speaker ID
    spk_threshold: float = 0.4
    # Режимы для Oracle (VAD, ASR, Diarization)
    use_oracle_vad : bool = False    # Если истинно, то у VAD включается режим Оракула
    use_oracle_asr: bool = False
    use_oracle_diarization: bool = False


# Создаем экземпляр (синглтон) для импорта в другие файлы
config = Config()
pl_conf = PipelineConfig(config = config)
