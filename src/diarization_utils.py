"""Модуль содержит утилиты для разных способов диаризации"""
from enum import Enum, auto
import model_utils
import asr_utils
import config

class SpeakerResolvingMode(Enum):
    """Возможные способы диаризации"""
    # Без диаризации
    NONE = auto()

    # Способы на основе VAD (посегментно)
    VAD_SPEAKER_MANAGER = auto()   # С использованием вашего менеджера и базы
    VAD_SIMPLE_THRESHOLD = auto()  # Простое сравнение эмбеддингов по порогу
    VAD_SPEECHBRAIN = auto()       # Использование моделей SpeechBrain для эмбеддингов

    # Полноценная диаризация (целым файлом или потоком)
    PYANNOTE_OFFLINE = auto()      # Обработка всего файла целиком
    PYANNOTE_STREAMING = auto()    # Потоковая диаризация

class SpeakerResolver:
    """Класс для определения спикера"""
    def __init__(self, num_threads, spk_threshold, resolving_mode: SpeakerResolvingMode = None):
        self._resolving_mode = resolving_mode
        self._num_threads = num_threads
        self._spk_threshold = spk_threshold
        self._spk_model = config.config.get_embedding_model()["model"]
        self._provider = config.config.get_work()["provider"]
        if self._resolving_mode == SpeakerResolvingMode.VAD_SPEAKER_MANAGER:
            self._extractor, self._manager = model_utils.load_embedder(
                self._spk_model, num_threads = num_threads, provider = self._provider
            )

            self._speaker_id = 0

    def resolve(self, seg):
        """
            Вычисляет эмбеддинг голоса из фразы.
            Пытается найти соответствие в базе голосов, если находит, то возвращает id спикера.
            Если голос не найден, то создается, сохраняется в базе и возвращается новый спикер.
        """
        # Speaker embedding + dynamic enrollment
        emb = asr_utils.compute_embedding(self._extractor, seg)
        name = self._manager.search(emb, threshold = self._spk_threshold)
        if not name:
            name = f"SPK_{self._speaker_id:02d}"
            ok = self._manager.add(name, emb)
            if not ok:
                raise RuntimeError(f"Failed to register speaker {name}")
            self._speaker_id += 1
        # ToDo: перейти на ID спикеров вместо имен
        return name
