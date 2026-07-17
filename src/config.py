"""
Модуль загрузки конфигурации
"""
from dataclasses import dataclass
from pathlib import Path
import os
from enum import Enum, auto
from typing import Any
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

AUDIO_PATH_MIC = "mic"
AUDIO_PATH_ORACLE_EMPTY = "oracle:empty"

class PipelineType(Enum):
    """Содержит типы пайплайнов"""
    ASR_PIPELINE = "asr"
    MANAGER_DIARIZ_PIPELINE = "dman"
    CENTRIOD_DIARIZ_PIPELINE = "dcentr"

class SpeakerResolvingMode(Enum):
    """Возможные способы диаризации"""
    NONE = auto()       # Без диаризации
    ORACLE = auto()     # Режим Оракула
    VAD_SPEAKER_MANAGER = auto()   # С использованием менеджера и базы
    VAD_SIMPLE_CENTROID = auto()   # Cравнение эмбеддингов с центроидами по порогу

class SpeakerRepoType(Enum):
    """Варианты репозитария спикеров"""
    IN_MEMORY = "in_memory"
    DB_SQLITE = "db_sqlite"

class Config:
    """Класс загрузчика конфигурации"""
    def __init__(self) -> None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        # Определяем рабочие модели
        self.vad_model_name = self._data["work"]["vad_model_name"]
        self.embed_model_name = self._data["work"]["embed_model_name"]
        self.asr_model_name = self._data["work"]["asr_model_name"]
        self.segmentation_model_name = self._data["work"]["segmentation_model_name"]
        self.provider = self._data["work"]["provider"]

    def get_profiles(self) -> Any:
        """Возвращает список профилей и их настроек"""
        return self._data.get("profiles", {})

    def get_new_runtime_config(self)-> RuntimeConfig:
        """Создает рантайм конфиг"""
        return RuntimeConfig(
            provider = self.provider,
        )

    def get_new_vad_config(self, model_name: str | None = None)-> VadConfig:
        """Создает конфиг для VAD модели"""
        if model_name is None: # Если имя модели не указано, берем имя по умолчанию
            model_name = self.vad_model_name
        model_data = self._data["models"]["vad"][model_name]
        return VadConfig(
            model_name = model_name,
            model_short_name = model_data["short_name"],
            model_path = model_data["model"],
        )

    def get_new_asr_config(self, model_name: str | None = None)-> AsrConfig:
        """Создает конфиг для ASR модели"""
        if model_name is None: # Если имя модели не указано, берем имя по умолчанию
            model_name = self.asr_model_name
        model_data = self._data["models"]["asr"][model_name]
        return AsrConfig(
            model_name = model_name,
            model_short_name = model_data["short_name"],
            model_type = model_data["asr_type"],
            # NeMo CTC
            nemo_model_path = model_data.get("model", None),
            nemo_tokens_path = model_data.get("tokens", None),
            # Qwen3
            qwen3_conv_frontend_path = model_data.get("conv_frontend", None),
            qwen3_encoder_path = model_data.get("encoder", None),
            qwen3_decoder_path = model_data.get("decoder", None),
            qwen3_tokenizer_path = model_data.get("tokenizer_dir", None),
        )

    def get_new_embedding_config(self, model_name: str | None = None)-> EmbeddingConfig:
        """Создает конфиг для эмбеддинг модели"""
        if model_name is None: # Если имя модели не указано, берем имя по умолчанию
            model_name = self.embed_model_name
        model_data = self._data["models"]["embedding"][model_name]
        return EmbeddingConfig(
            model_name = model_name,
            model_short_name = model_data["short_name"],
            model_path = model_data["model"],
        )

    def get_new_diar_vad_config(self)-> VadDiarizationConfig:
        """Создает конфиг для VAD диаризации"""
        return VadDiarizationConfig()

    def get_new_segmentation_config(self, model_name: str | None = None)-> SegmentationConfig:
        """Создает конфиг для модели pyannote"""
        if model_name is None: # Если имя модели не указано, берем имя по умолчанию
            model_name = self.segmentation_model_name
        model_data = self._data["models"]["segmentation"][model_name]
        return SegmentationConfig(
            model_name = model_name,
            model_short_name = model_data["short_name"],
            model_path = model_data["model"],
        )

    def get_new_pipeline_config(
        self,
        vad_model_name: str | None = None,
        asr_model_name: str | None = None,
        embed_model_name: str | None = None,
        segmentation_model_name: str | None = None,
    ) -> PipelineConfig:
        """Создает конфиг для пайплайна"""
        return PipelineConfig(
            runtime = config.get_new_runtime_config(),
            vad = config.get_new_vad_config(model_name = vad_model_name),
            asr = config.get_new_asr_config(model_name = asr_model_name),
            embed = config.get_new_embedding_config(model_name = embed_model_name),
            diar_vad = config.get_new_diar_vad_config(),
            segmentation = config.get_new_segmentation_config(model_name = segmentation_model_name),
        )

@dataclass
class RuntimeConfig:
    """Общие настройки окружения и выполнения"""
    pipeline_type: PipelineType = PipelineType.ASR_PIPELINE
    num_threads: int = 1
    provider: str = "cpu"
    output_dir: str = str(DEFAULT_OUTPUT_DIR)    # Путь к папке с с файлами с распознанным текстом
    no_timestamps: bool = False             # Запрещает вывод меток времени в распознанный текст
    use_db: bool = False    # Признак хранения спикеров, аудиофайлов и сегментов в БД или в памяти

@dataclass
class VadConfig:
    """Настройки модели Voice Activity Detection (обнаружение речи)"""
    use_oracle: bool = False
    model_name: str | None = None
    model_short_name: str | None = None
    model_path: str | None = None
    threshold: float = 0.3 # 0.4 Было 0.3
    min_silence: float = 0.5 # 0.1 Было 0.25
    min_speech: float = 0.1 # 0.2 Было 0.25
    max_speech: float = 30.0

@dataclass
class AsrConfig:
    """Настройки модели распознавания речи (ASR)"""
    use_oracle: bool = False
    model_name: str | None = None
    model_short_name: str | None = None
    model_type: str | None = None
    # NeMo CTC
    nemo_model_path: str | None = None
    nemo_tokens_path: str | None = None
    # Qwen3
    qwen3_conv_frontend_path: str | None = None
    qwen3_encoder_path: str | None = None
    qwen3_decoder_path: str | None = None
    qwen3_tokenizer_path: str | None = None

@dataclass
class EmbeddingConfig:
    """Настройки модели эмбеддинга"""
    use_oracle: bool = False
    model_name: str | None = None
    model_short_name: str | None = None
    model_path: str | None = None

@dataclass
class VadDiarizationConfig:
    """Настройки VAD диаризации"""
    use_oracle: bool = False
    spk_threshold: float = 0.4  # Порог косинусной схожести для поиска спикера
    resolving_mode: SpeakerResolvingMode = SpeakerResolvingMode.NONE
    speaker_repo_type: SpeakerRepoType = SpeakerRepoType.IN_MEMORY
    db_path: str = str(DB_DEFAULT_PATH)

@dataclass
class SegmentationConfig:
    """Настройки модели сегментации"""
    use_oracle: bool = False
    model_name: str | None = None
    model_short_name: str | None = None
    model_path: str | None = None
    spk_threshold: float = 0.4
    min_seg_sec: float = 0.5
    num_speakers: int = -1
    cluster_threshold: float = 0.5
    pad_sec: float = 0.2
    merge_gap: float = 0.25
    min_turn_sec: float = 0.6
    show_progress: bool = False

@dataclass
class PipelineConfig:
    """Главный конфигурационный класс пайплайна"""
    runtime: RuntimeConfig
    vad: VadConfig
    asr: AsrConfig
    embed: EmbeddingConfig
    diar_vad: VadDiarizationConfig
    segmentation: SegmentationConfig

# Создаем экземпляр (синглтон) для импорта в другие файлы
config = Config()
pl_conf = config.get_new_pipeline_config()
