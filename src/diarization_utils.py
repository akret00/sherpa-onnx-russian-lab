"""Модуль содержит утилиты для разных способов диаризации"""
from enum import Enum, auto
from dataclasses import dataclass
import numpy
import model_utils
import asr_utils
import config
import segment_utils
from speaker_storage import Speaker

@dataclass
class ResolveResult:
    """Содержит результаты определения спикера"""
    speaker: Speaker | None = None
    cos_similarity: float = -1  # косинусная схожесть или другой скор

class SpeakerResolvingMode(Enum):
    """Возможные способы диаризации"""
    # Без диаризации
    NONE = auto()
    # Способы на основе VAD (посегментно)
    VAD_SPEAKER_MANAGER = auto()   # С использованием вашего менеджера и базы
    VAD_SIMPLE_THRESHOLD = auto()  # Простое сравнение эмбеддингов по порогу
    VAD_SIMPLE_CENTROID = auto()   # Простое сравнение эмбеддингов с центроидами по порогу
    VAD_SPEECHBRAIN = auto()       # Использование моделей SpeechBrain для эмбеддингов
    # Полноценная диаризация (целым файлом или потоком)
    PYANNOTE_OFFLINE = auto()      # Обработка всего файла целиком
    PYANNOTE_STREAMING = auto()    # Потоковая диаризация

class SpeakerResolver:
    """Класс для определения спикера"""
    def __init__(
            self,
            num_threads: int,
            spk_threshold: float,
            resolving_mode: SpeakerResolvingMode = None,
            speakers: list[Speaker] | None = None,
    ):
        self._resolving_mode = resolving_mode
        self._num_threads = num_threads
        # spk_threshold -  косинусное сходство: Обычно лежит в диапазоне 0.0 (совсем разные)
        # до 1.0 (идентичные).
        # Норма примерно 0.5 - 0.6, если есть похожие голоса, то нужно повышать до 0.65+
        # если шум, эхо, порог придется снижать, но тогда могут дробиться реальные спикеры
        self._spk_threshold = spk_threshold
        self._spk_model = config.pl_conf.embedding_model_path
        self._provider = config.pl_conf.provider
        # Список спикеров с векторами и количеством накопленных фраз
        if speakers is None:
            self._speakers: list[Speaker] = []
        else:
            self._speakers: list[Speaker] = speakers

        if self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTROID:
            # Переменные для сброса последнего спикера при паузах
            self._last_spk = None
            self._last_end_time = 0.0

        # Инициализация экстрактора
        if self._resolving_mode in (
            SpeakerResolvingMode.VAD_SPEAKER_MANAGER,
            SpeakerResolvingMode.VAD_SIMPLE_CENTROID,
        ):
            self._extractor, self._manager = model_utils.load_embedder(
                self._spk_model, num_threads = self._num_threads, provider = self._provider
            )

            if self._resolving_mode == SpeakerResolvingMode.VAD_SPEAKER_MANAGER:
                # Загружаем базу спикеров в менеджер
                for index, spk in enumerate(self._speakers):
                    self._manager.add(str(index + 1), spk.embedding)
        elif self._resolving_mode == SpeakerResolvingMode.NONE:
            pass # Ничего не делаем
        else:
            raise ValueError(f"Тип диаризации {resolving_mode} пока не поддерживается")

    def _normalize_vector(self, vec):
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            return vec  # Защита от деления на ноль, если вектор пустой
        return vec / norm

    def _update_speaker_profile(self, speaker: Speaker, new_emb, alpha=0.1):
        """Мягкое обновление центроида спикера"""
        old_centroid = speaker.embedding
        # Формула экспоненциального сглаживания
        updated_centroid = (1 - alpha) * old_centroid + alpha * new_emb
        # Нормализуем вектор обратно (важно для косинусного сходства)
        speaker.embedding = self._normalize_vector(updated_centroid)

    def _search_or_create_speaker_manager(self, emb) -> ResolveResult:
        matched_id = self._manager.search(emb, threshold = self._spk_threshold)
        if not matched_id:
            spk = Speaker(
                name = f"SPK_{(len(self._speakers) + 1):03d}",
                embedding = emb,
            )

            self._speakers.append(spk)
            matched_id = str((len(self._speakers)))
            ok = self._manager.add(matched_id, emb)
            if not ok:
                raise RuntimeError(f"Failed to register speaker {matched_id}")

        spk = self._speakers[int(matched_id) - 1]
        score = numpy.dot(spk.embedding, emb) # Косинусное для нормированных векторов

        return ResolveResult(speaker = spk, cos_similarity = score)

    def _search_or_create_speaker_centriod(self, seg, emb) -> ResolveResult:
        """
        Делает:
            - поиск эмбеддинга фразы спикера среди центроидов
            - если спикер не найден, а фраза качественная, то добавляет новый центроид
            - если фраза качественная, и спикер найден, то ообновляет его центроид
        """
        # Ищем в нашей базе через косинусное сходство
        best_spk = None
        best_score = -1
        for curr_spk in self._speakers:
            score = numpy.dot(curr_spk.embedding, emb) # Косинусное для нормированных векторов
            if score > best_score:
                best_score = score
                best_spk = curr_spk

        # print(f"Threshold: {self._spk_threshold} Best score: {best_score}")

        if best_score > self._spk_threshold:
            spk = best_spk
            # Обновляем профиль только если фраза длинная (> 2 сек) = качественная
            if len(seg) > 2.0 * config.SR:
                self._update_speaker_profile(spk, emb)
        else:
            # Создаем нового и добавляем к базе
            spk = Speaker(
                name = f"SPK_{(len(self._speakers) + 1):03d}",
                embedding = emb,
            )

            self._speakers.append(spk)
            best_score = 1.0

        return ResolveResult(speaker = spk, cos_similarity = best_score)

    def get_speakers(self):
        """Возвращает список спикеров"""
        return self._speakers

    def resolve(self, seg, t_start = 0, t_end = 0):
        """
            Вычисляет эмбеддинг голоса из фразы.
            Пытается найти соответствие в базе голосов, если находит, то возвращает id спикера.
            Если голос не найден, то создается, сохраняется в базе и возвращается новый спикер.
        """
        # ToDo: сделать более похожим алгоритм для обоих способов определения спикеров
        # Вернуть пустой ResolveResult, если режим SpeakerResolvingMode.NONE
        if self._resolving_mode == SpeakerResolvingMode.NONE:
            resolve_result = ResolveResult(speaker = None, cos_similarity = -1)
        # Расчет эмбеддинга спикера и поиск спикера по эмбеддингам
        elif self._resolving_mode == SpeakerResolvingMode.VAD_SPEAKER_MANAGER:
            emb = self._normalize_vector(asr_utils.compute_embedding(self._extractor, seg))
            resolve_result = self._search_or_create_speaker_manager(emb)
            resolve_result.speaker.count += 1
            resolve_result.speaker.total_count += 1

        # Расчет эмбеддинга спикера и поиск спикера по центроидам эмбеддингов
        elif self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTROID:
            # 1. Сброс инерции при длинной паузе
            pause_duration = t_start - self._last_end_time
            if pause_duration > config.MAX_PAUSE_FOR_INERTIA:
                self._last_spk = None
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
                resolve_result = self._search_or_create_speaker_centriod(vad_seg, emb)
                resolve_result.speaker.count += 1
                resolve_result.speaker.total_count += 1
                self._last_spk = resolve_result.speaker # Обновляем "уверенного" спикера
            else:
                # Сегмент короткий: берем последнего или Unknown
                resolve_result = ResolveResult(
                    speaker = self._last_spk,
                    cos_similarity = -1,
                )

        return resolve_result
