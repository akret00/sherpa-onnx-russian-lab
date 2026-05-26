"""Модуль содержит утилиты для разных способов диаризации"""
from enum import Enum, auto
import numpy
import model_utils
import asr_utils
import config
import segment_utils

class SpeakerResolvingMode(Enum):
    """Возможные способы диаризации"""
    # Без диаризации
    NONE = auto()

    # Способы на основе VAD (посегментно)
    VAD_SPEAKER_MANAGER = auto()   # С использованием вашего менеджера и базы
    VAD_SIMPLE_THRESHOLD = auto()  # Простое сравнение эмбеддингов по порогу
    VAD_SIMPLE_CENTRIOD = auto()   # Простое сравнение эмбеддингов с центроидами по порогу
    VAD_SPEECHBRAIN = auto()       # Использование моделей SpeechBrain для эмбеддингов

    # Полноценная диаризация (целым файлом или потоком)
    PYANNOTE_OFFLINE = auto()      # Обработка всего файла целиком
    PYANNOTE_STREAMING = auto()    # Потоковая диаризация

class SpeakerResolver:
    """Класс для определения спикера"""
    def __init__(self, num_threads, spk_threshold, resolving_mode: SpeakerResolvingMode = None):
        self._resolving_mode = resolving_mode
        self._num_threads = num_threads
        # spk_threshold -  косинусное сходство: Обычно лежит в диапазоне 0.0 (совсем разные)
        # до 1.0 (идентичные).
        # Норма примерно 0.5 - 0.6, если есть похожие голоса, то нужно повышать до 0.65+
        # если шум, эхо, порог придется снижать, но тогда могут дробиться реальные спикеры
        self._spk_threshold = spk_threshold
        self._spk_model = config.config.get_embedding_model()["model"]
        self._provider = config.config.get_work()["provider"]

        if self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTRIOD:
            # Переменные для сброса последнего спикера при паузах
            self._last_name = "Unknown"
            self._last_end_time = 0.0
            # Словарь спикеров с векторами и количеством накопленных фраз
            # { "SPK_01": {"centroid": np_array, "count": int, "label": "Иван"} }
            self._speaker_db = {}

        # Инициализация экстрактора
        if self._resolving_mode in (
            SpeakerResolvingMode.VAD_SPEAKER_MANAGER,
            SpeakerResolvingMode.VAD_SIMPLE_CENTRIOD,
        ):
            self._extractor, self._manager = model_utils.load_embedder(
                self._spk_model, num_threads = self._num_threads, provider = self._provider
            )

            self._speaker_id = 0
        else:
            raise ValueError(f"Тип диаризации {resolving_mode} пока не поддерживается")

    def _normalize_vector(self, vec):
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            return vec  # Защита от деления на ноль, если вектор пустой
        return vec / norm

    def _update_speaker_profile(self, name, new_emb, alpha=0.1):
        """Мягкое обновление центроида спикера"""
        old_centroid = self._speaker_db[name]["centroid"]
        # Формула экспоненциального сглаживания
        updated_centroid = (1 - alpha) * old_centroid + alpha * new_emb
        # Нормализуем вектор обратно (важно для косинусного сходства)
        self._speaker_db[name]["centroid"] = self._normalize_vector(updated_centroid)
        self._speaker_db[name]["count"] += 1

    def _search_speaker(self, seg, emb):
        """
        Делает:
            - поиск эмбеддинга фразы спикера среди центроидов
            - если спикер не найден, а фраза качественная, то добавляет новый центроид
            - если фраза качественная, и спикер найден, то ообновляет его центроид
        """
        # Ищем в нашей базе через косинусное сходство
        best_name = None
        best_score = -1
        for curr_name, profile in self._speaker_db.items():
            score = numpy.dot(profile["centroid"], emb) # Косинусное для нормированных векторов
            if score > best_score:
                best_score = score
                best_name = curr_name

        # print(f"Threshold: {self._spk_threshold} Best score: {best_score}")

        if best_score > self._spk_threshold:
            name = best_name
            # Обновляем профиль только если фраза длинная (> 2 сек) = качественная
            if len(seg) > 2.0 * config.SR:
                self._update_speaker_profile(name, emb)
        else:
            # Создаем нового
            name = f"SPK_{self._speaker_id:02d}"
            self._speaker_db[name] = {
                "centroid": emb, 
                "count": 1, 
                "label": name
            }
            self._speaker_id += 1

        return name

    def resolve(self, seg, t_start = 0, t_end = 0):
        """
            Вычисляет эмбеддинг голоса из фразы.
            Пытается найти соответствие в базе голосов, если находит, то возвращает id спикера.
            Если голос не найден, то создается, сохраняется в базе и возвращается новый спикер.
        """
        # Расчет эмбеддинга спикера и поиск спикера по эмбеддингам
        if self._resolving_mode == SpeakerResolvingMode.VAD_SPEAKER_MANAGER:
            emb = self._normalize_vector(asr_utils.compute_embedding(self._extractor, seg))
            name = self._manager.search(emb, threshold = self._spk_threshold)
            if not name:
                name = f"SPK_{self._speaker_id:02d}"
                ok = self._manager.add(name, emb)
                if not ok:
                    raise RuntimeError(f"Failed to register speaker {name}")
                self._speaker_id += 1
        # Расчет эмбеддинга спикера и поиск спикера по центроидам эмбеддингов
        elif self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTRIOD:
            # 1. Сброс инерции при длинной паузе
            pause_duration = t_start - self._last_end_time
            if pause_duration > config.MAX_PAUSE_FOR_INERTIA:
                self._last_name = "Unknown"
            self._last_end_time = t_end # Делать для всех или только качественных сегментов?

            # 2. Обрезка правого края с тишиной для коротких фраз
            if len(seg) <= int(1.5 * config.SR):
                vad_seg = segment_utils.trim_silence_fix_end(seg)
                # segment_utils.visualize_segment_energy(vad_seg)
            else:
                vad_seg = seg

            # 3. Логика определения спикера (только для качественных сегментов)
            if len(vad_seg) >= int(config.MIN_SEARCH_SEG_LEN * config.SR):
                emb = self._normalize_vector(asr_utils.compute_embedding(self._extractor, vad_seg))
                name = self._search_speaker(vad_seg, emb)
                self._last_name = name # Обновляем "уверенного" спикера
            else:
                # Сегмент короткий: берем последнего или Unknown
                name = self._last_name

        # ToDo: перейти на ID спикеров вместо имен
        return name
