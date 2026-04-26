"""
Модуль загрузки конфигурации
"""
from pathlib import Path
import yaml

# Частота дискретизации
SR = 16000

class Config:
    """Класс загрузчика конфигурации"""
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        # Определяем рабочие модели
        self._vad_model_name = self._data['work']['vad_model_name']
        self._embed_model_name = self._data['work']['embed_model_name']
        self._asr_model_name = self._data['work']['asr_model_name']

    # @property
    def get_main(self):
        """Возвращает базовые настройки"""
        return self._data.get("main", {})

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

# Создаем экземпляр (синглтон) для импорта в другие файлы
config = Config()
