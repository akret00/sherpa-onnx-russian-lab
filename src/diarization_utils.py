"""Модуль содержит утилиты для разных способов диаризации"""
from dataclasses import dataclass
import typing
import numpy
import sherpa_onnx
import model_utils
from config import (
    SR, MIN_SEARCH_SEG_LEN, MAX_PAUSE_FOR_INERTIA, PipelineConfig, SpeakerResolvingMode
)
import segment_utils
from entities import Speaker, AudioSegment
from speaker_storage import BaseRepo, create_spk_repo

def compute_embedding(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    samples_f32: numpy.ndarray
) -> numpy.ndarray:
    """Рассчитывает эмбеддинг голоса для сегмента аудио"""
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate = SR, waveform = samples_f32)
    stream.input_finished()
    # extractor.is_ready(stream) обычно True, если сегмент не слишком короткий
    emb = extractor.compute(stream)
    return numpy.array(emb, dtype=numpy.float32)

@dataclass
class ResolveResult:
    """Содержит результаты определения спикера"""
    speaker: Speaker | None = None
    cos_similarity: float = -1  # косинусная схожесть или другой скор

class SpeakerResolver:
    """Класс для определения спикера"""
    def __init__(self, pl_conf: PipelineConfig):
        self._pl_conf: PipelineConfig = pl_conf
        # Автоматическое включение режима Оракула
        if self._pl_conf.diar_vad.use_oracle:
            self._resolving_mode = SpeakerResolvingMode.ORACLE
        else:
            self._resolving_mode = self._pl_conf.diar_vad.resolving_mode

        self._num_threads = self._pl_conf.runtime.num_threads
        # spk_threshold -  косинусное сходство: Обычно лежит в диапазоне 0.0 (совсем разные)
        # до 1.0 (идентичные).
        # Норма примерно 0.5 - 0.6, если есть похожие голоса, то нужно повышать до 0.65+
        # если шум, эхо, порог придется снижать, но тогда могут дробиться реальные спикеры
        self._spk_threshold = self._pl_conf.diar_vad.spk_threshold
        self._embed_model_path = self._pl_conf.embed.model_path
        self._provider = self._pl_conf.runtime.provider

        # Создаем репо спикеров
        self._spk_repo: BaseRepo = create_spk_repo(pl_conf = self._pl_conf)
        # Список спикеров с векторами и количеством накопленных фраз
        self._speakers: list[Speaker] = self._spk_repo.load_speakers()

        if self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTROID:
            # Переменные для сброса последнего спикера при паузах
            self._last_spk = None
            self._last_end_time = 0.0

        # Эталонная разметка для Оракула
        self.markup_segments: list[AudioSegment] = []
        self.current_markup_segment_num = 0
        self.padding = 0

        # Инициализация экстрактора
        if self._resolving_mode in (
            SpeakerResolvingMode.VAD_SPEAKER_MANAGER,
            SpeakerResolvingMode.VAD_SIMPLE_CENTROID,
        ):
            self._extractor, self._manager = model_utils.load_embedder(
                self._embed_model_path, num_threads = self._num_threads, provider = self._provider
            )

            if self._resolving_mode == SpeakerResolvingMode.VAD_SPEAKER_MANAGER:
                # Загружаем базу спикеров в менеджер
                for index, spk in enumerate(self._speakers):
                    self._manager.add(
                        str(index + 1),
                        spk.get_embedding(model_name = self._pl_conf.embed.model_short_name)
                    )
        elif self._resolving_mode == SpeakerResolvingMode.NONE:
            pass # Ничего не делаем
        elif self._resolving_mode == SpeakerResolvingMode.ORACLE:
            pass # Ничего не делаем
        else:
            raise ValueError(f"Тип диаризации {self._resolving_mode} пока не поддерживается")

    def _normalize_vector(self, vec: numpy.ndarray) -> numpy.ndarray:
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            return vec  # Защита от деления на ноль, если вектор пустой
        return typing.cast(numpy.ndarray, vec / norm)

    def _update_speaker_profile(
        self, speaker: Speaker, new_emb: numpy.ndarray, alpha: float =0.1
    ) -> None:
        """Мягкое обновление центроида спикера"""
        old_centroid = speaker.get_embedding(model_name = self._pl_conf.embed.model_short_name)
        if old_centroid is None:
            raise ValueError("Эмбеддинг не может иметь значение None")
        # Формула экспоненциального сглаживания
        updated_centroid = (1 - alpha) * old_centroid + alpha * new_emb
        # Нормализуем вектор обратно (важно для косинусного сходства)
        speaker.add_embedding(
            model_name = self._pl_conf.embed.model_short_name,
            embedding = self._normalize_vector(updated_centroid)
        )

    def _search_or_create_speaker_manager(self, emb: numpy.ndarray) -> ResolveResult:
        matched_id = self._manager.search(emb, threshold = self._spk_threshold)
        if not matched_id:
            # Создает пустого спикера и добавляем к нему эмбеддинг с именем модели
            spk = Speaker()
            spk.add_embedding(model_name = self._pl_conf.embed.model_short_name, embedding = emb)
            # При сохранении пустого спикера в репо, ему будут присвоены ИД и имя по умолчанию
            self._spk_repo.save_speakers(speakers = [spk])
            # Добавляем нового спикера в локальный список
            self._speakers.append(spk)

            matched_id = str((len(self._speakers)))
            if not self._manager.add(matched_id, emb):
                raise RuntimeError(f"Failed to register speaker {matched_id}")

        spk = self._speakers[int(matched_id) - 1]
        score = numpy.dot(
            spk.get_embedding(model_name = self._pl_conf.embed.model_short_name),
            emb
        ) # Косинусное сходство для нормированных векторов

        return ResolveResult(speaker = spk, cos_similarity = float(score))

    def _search_or_create_speaker_centriod(
        self, seg: numpy.ndarray, emb: numpy.ndarray
    ) -> ResolveResult:
        """
        Делает:
            - поиск эмбеддинга фразы спикера среди центроидов
            - если спикер не найден, а фраза качественная, то добавляет новый центроид
            - если фраза качественная, и спикер найден, то ообновляет его центроид
        """
        # Ищем в нашей базе через косинусное сходство
        best_spk = None
        best_score = -1.0
        for curr_spk in self._speakers:
            score = numpy.dot(
                curr_spk.get_embedding(model_name = self._pl_conf.embed.model_short_name),
                emb
            ) # Косинусное сходство для нормированных векторов
            if score > best_score:
                best_score = score
                best_spk = curr_spk

        # print(f"Threshold: {self._spk_threshold} Best score: {best_score}")

        if best_score > self._spk_threshold:
            spk = best_spk
            # Обновляем профиль только если фраза длинная (> 2 сек) = качественная
            if len(seg) > 2.0 * SR:
                self._update_speaker_profile(spk, emb)
        else:
            # Создает пустого спикера и добавляем к нему эмбеддинг с именем модели
            spk = Speaker()
            spk.add_embedding(model_name = self._pl_conf.embed.model_short_name, embedding = emb)
            # При сохранении пустого спикера в репо, ему будут присвоены ИД и имя по умолчанию
            self._spk_repo.save_speakers(speakers = [spk])
            # Добавляем нового спикера в локальный список
            self._speakers.append(spk)

            best_score = 1.0

        return ResolveResult(speaker = spk, cos_similarity = float(best_score))

    def get_speakers(self) -> list[Speaker]:
        """Возвращает список спикеров"""
        return self._speakers

    def resolve(self, seg: numpy.ndarray, t_start: float = 0, t_end: float = 0) -> ResolveResult:
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
            emb = self._normalize_vector(compute_embedding(self._extractor, seg))
            resolve_result = self._search_or_create_speaker_manager(emb)
            resolve_result.speaker.count += 1
            resolve_result.speaker.total_count += 1

        # Расчет эмбеддинга спикера и поиск спикера по центроидам эмбеддингов
        elif self._resolving_mode == SpeakerResolvingMode.VAD_SIMPLE_CENTROID:
            # 1. Сброс инерции при длинной паузе
            pause_duration = t_start - self._last_end_time
            if pause_duration > MAX_PAUSE_FOR_INERTIA:
                self._last_spk = None
            self._last_end_time = t_end # Делать для всех или только качественных сегментов?

            # 2. Обрезка правого края с тишиной для коротких фраз
            if len(seg) <= int(1.5 * SR):
                vad_seg = segment_utils.trim_silence_fix_end(seg)
                # segment_utils.visualize_segment_energy(vad_seg)
            else:
                vad_seg = seg

            # 3. Логика определения спикера (только для качественных сегментов)
            if len(vad_seg) >= int(MIN_SEARCH_SEG_LEN * SR):
                emb = self._normalize_vector(compute_embedding(self._extractor, vad_seg))
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

        # Определение спикера на основе эталонной разметки
        elif self._resolving_mode == SpeakerResolvingMode.ORACLE:
            if t_start is None:
                raise ValueError("Не задано значение аргумента t_start")

            collected_speakers = []

            # Итерируемся по сегментам, пока не упремся в конец списка или в будущее время
            while self.current_markup_segment_num < len(self.markup_segments):
                orc_seg = self.markup_segments[self.current_markup_segment_num]

                # Если сегмент уже должен был прозвучать
                if t_start >= orc_seg.start_time - self.padding:
                    if orc_seg.speaker_id:  # Защита от None или пустых строк
                        collected_speakers.append(Speaker(
                            id = orc_seg.speaker_id,
                            name = f"SPK_{orc_seg.speaker_id:03d}",
                            )
                        )
                    self.current_markup_segment_num += 1
                else:
                    # Сегмент из будущего, прекращаем сбор для текущего вызова
                    break

            # В локальную базу спикеров не добавляем, в режиме Оракула она остается пустой
            if collected_speakers:
                resolve_result = ResolveResult(speaker = collected_speakers[0], cos_similarity = 1)
            else:
                resolve_result = ResolveResult(
                    speaker = Speaker(
                        id = -1000,
                        name = "SPK_BAD"
                    ),
                    cos_similarity = 1
                )

        return resolve_result

    def set_markup_segments(self, markup_segments: list[AudioSegment]) -> None:
        """Специфичный метод Оракула для загрузки ручной разметки фраз."""
        # Список markup_segments уже должен быть отсортирован по времени начала сегмента
        # и проверен на отрицательную длительность сегментов
        self.markup_segments = markup_segments

    def reset(self) -> None:
        """Сброс состояния Оракула перед новым циклом"""
        self.current_markup_segment_num = 0
        self._speakers = [] # Очищаем список спикеров
