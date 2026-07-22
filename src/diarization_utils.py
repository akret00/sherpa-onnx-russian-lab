"""Модуль содержит утилиты для разных способов диаризации"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import typing
import numpy
import sherpa_onnx
from config import (
    SR, MIN_SEARCH_SEG_LEN, PipelineConfig, SpeakerResolvingMode
)
import model_utils
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

class SpeakerResolver(ABC):
    """Базовый класс для определения спикера"""
    @abstractmethod
    def resolve(self, seg: numpy.ndarray, t_start: float = 0, t_end: float = 0) -> ResolveResult:
        """Возвращает объект ResolveResult"""

    @abstractmethod
    def get_speakers(self) -> list[Speaker]:
        """Возвращает список спикеров"""

    @abstractmethod
    def save_all_speakers(self) -> None:
        """Сохраняет в репозиторий обновленные данные всех локальных спикеров"""


class BasePassiveSpeakerResolver(SpeakerResolver, ABC):
    """Базовый класс для имитации определения спикера"""

    def get_speakers(self) -> list[Speaker]:
        """Возвращает пустой список спикеров"""
        return []

    def save_all_speakers(self) -> None:
        """Сохраняет в репозиторий обновленные данные всех локальных спикеров"""

class NoneSpeakerResolver(BasePassiveSpeakerResolver):
    """Класс - заглушка для определения спикера в режиме только ASR"""
    def resolve(self, seg: numpy.ndarray, t_start: float = 0, t_end: float = 0) -> ResolveResult:
        """Возвращает пустой объект ResolveResult"""
        return ResolveResult(speaker = None, cos_similarity = -1)


class OracleSpeakerResolver(BasePassiveSpeakerResolver):
    """Класс для определения спикера на базе Оракула"""
    def __init__(self, pl_conf: PipelineConfig):
        self._pl_conf: PipelineConfig = pl_conf

        # Эталонная разметка для Оракула
        self.markup_segments: list[AudioSegment] = []
        self.current_markup_segment_num = 0
        self.padding = 0

    def resolve(self, seg: numpy.ndarray, t_start: float = 0, t_end: float = 0) -> ResolveResult:
        """Имитирует определение спикера на базе эталонной разметки"""
        # Определение спикера на основе эталонной разметки
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


class VadSpeakerResolver(SpeakerResolver):
    """Класс для определения спикера на базе центроидов"""
    def __init__(self, pl_conf: PipelineConfig):
        self._pl_conf: PipelineConfig = pl_conf

        self._num_threads = self._pl_conf.runtime.num_threads
        # _spk_threshold -  косинусное сходство: Обычно лежит в диапазоне 0.0 (совсем разные)
        # до 1.0 (идентичные).
        # Норма примерно 0.5 - 0.6, если есть похожие голоса, то нужно повышать до 0.65+
        # если шум, эхо, порог придется снижать, но тогда могут дробиться реальные спикеры
        self._spk_threshold = self._pl_conf.embed.threshold
        self._embed_model_path = self._pl_conf.embed.model_path
        self._provider = self._pl_conf.runtime.provider
        self._model_name = self._pl_conf.embed.model_short_name

        # Инициализация экстрактора эмбеддингов
        self._extractor, self._manager = model_utils.load_embedder(
            self._embed_model_path, num_threads = self._num_threads, provider = self._provider
        )

        # Создаем репо спикеров
        self._spk_repo: BaseRepo = create_spk_repo(pl_conf = self._pl_conf)

        # Готовим пустые структуры данных под кэш
        self._speakers_dict: dict[str, Speaker] = {}    # Словарь спикеров
        self._matrix: numpy.ndarray         # Матрица эмбеддингов
        self._active_ids: list[str] = []    # Список ИД спикеров в порядке эмбеддингов в матрице
        self._init_speaker_list()

    def _init_speaker_list(self) -> None:
        """
        - Загружает список спикеров из репо
        - Заполняет словарь спикеров спикерами, у которых есть эмбединги для текущей модели
        - Создает и наполняет матрицу эмбеддингов, а так же 
        """
        embeddings_list = []

        # Загружаем список спикеров с векторами и количеством накопленных фраз из репо
        speakers_list = self._spk_repo.load_speakers()

        # Фильтруем спикеров по наличию эмбеддинга для текущей модели и собираем данные для NumPy
        for spk in speakers_list:
            # Проверяем базовое наличие ID
            # if not hasattr(spk, 'id'):
            #     continue

            # Проверяем наличие эмбеддинга именно для ТЕКУЩЕЙ модели
            curr_emb = spk.get_embedding(model_name = self._model_name)
            if curr_emb is None:
                continue  # Пропускаем спикера, если он записан для другой модели

            spk_id = str(spk.id)

            # Заполняем словарь только валидными спикерами
            self._speakers_dict[spk_id] = spk
            # Синхронно наполняем списки для кэш-матрицы
            self._active_ids.append(spk_id)
            embeddings_list.append(curr_emb)

        # Инициализируем матрицу эмбеддингов на основе отфильтрованных данных
        if embeddings_list:
            self._matrix = numpy.vstack(embeddings_list)
        else:
            # Если подходящих спикеров нет, оставляем пустую матрицу для ленивой инициализации
            self._matrix = numpy.empty((0, 0), dtype=numpy.float32)

    def _normalize_vector(self, vec: numpy.ndarray) -> numpy.ndarray:
        """Выполняет нормализацию вектора"""
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            return vec  # Защита от деления на ноль, если вектор пустой
        return typing.cast(numpy.ndarray, vec / norm)

    def _create_and_register_speaker(self, emb: numpy.ndarray) -> Speaker:
        """Создает спикера, сохраняет в репо, и расширяет матрицу кэша."""
        spk = Speaker()
        spk.add_embedding(model_name = self._model_name, embedding = emb)
        self._spk_repo.save_speakers(speakers = [spk])

        spk_id = str(spk.id)
        self._speakers_dict[spk_id] = spk

        # Расширяем матрицу кэша новой строкой
        if self._matrix.shape[1] == 0:
            # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: если матрица пустая по второй оси, 
            # задаем её размерность на основе пришедшего вектора
            self._matrix = emb.reshape(1, -1)  # Делаем из вектора (D,) матрицу (1, D)
        else:
            self._matrix = numpy.vstack([self._matrix, emb])

        # Запоминаем, какой строке матрицы соответствует этот спикер
        self._active_ids.append(spk_id)

        return spk

    def _update_speaker_profile(
        self,
        speaker: Speaker,
        new_emb: numpy.ndarray,
        matrix_row_idx: int
    ) -> None:
        """Мягкое обновление центроида с точечной перезаписью в кэш-матрице."""
        old_centroid = speaker.get_embedding(model_name = self._model_name)
        if old_centroid is None:
            raise ValueError(
                f"Эмбеддинг спикера с ID: {speaker.id} для модели {self._model_name} "
                "не может иметь значение None"
            )

        # Динамический рассчет alpha
        alpha = self._pl_conf.diar_vad.min_alfa
        updated_centroid = (1.0 - alpha) * old_centroid + alpha * new_emb
        normalized_emb = self._normalize_vector(updated_centroid)

        # Обновляем объект спикера
        speaker.add_embedding(model_name = self._model_name, embedding = normalized_emb)

        # ОПТИМИЗАЦИЯ: Точечно меняем строку в существующей матрице без пересборки!
        self._matrix[matrix_row_idx] = normalized_emb

    def _search_or_create_speaker(self, seg: numpy.ndarray, emb: numpy.ndarray) -> ResolveResult:
        """
        Делает:
            - поиск эмбеддинга фразы спикера среди центроидов
            - если спикер не найден, а фраза качественная, то добавляет нового спикера
            - если фраза качественная, и спикер найден, то ообновляет его центроид
        """
        # Если база пуста — создаем первого спикера и возвращаем его
        if not self._active_ids:
            spk = self._create_and_register_speaker(emb)
            return ResolveResult(speaker=spk, cos_similarity=1.0)

        # Умножаем готовую матрицу из кэша на эмбединг, вместо перебора по строкам в цикле
        scores = numpy.dot(self._matrix, emb)

        best_index = int(numpy.argmax(scores))
        best_score = float(scores[best_index])
        best_spk_id = self._active_ids[best_index]
        best_spk = self._speakers_dict[best_spk_id]

        if best_score > self._spk_threshold:
            # Передаем best_index (номер строки в матрице), чтобы обновить её точечно
            if len(seg) > 2.5 * SR:  # Подняли порог длины до вашего нового значения
                self._update_speaker_profile(best_spk, emb, matrix_row_idx = best_index)
            return ResolveResult(speaker = best_spk, cos_similarity = best_score)

        # Создаем нового спикера
        spk = self._create_and_register_speaker(emb)
        return ResolveResult(speaker = spk, cos_similarity = 1.0)

    def get_speakers(self) -> list[Speaker]:
        """Возвращает копию списка спикеров"""
        return list(self._speakers_dict.values())

    def save_all_speakers(self) -> None:
        """Сохраняет в репозиторий обновленные данные всех локальных спикеров"""
        self._spk_repo.save_speakers(self.get_speakers())

    def resolve(self, seg: numpy.ndarray, t_start: float = 0, t_end: float = 0) -> ResolveResult:
        """
            Вычисляет эмбеддинг голоса из фразы.
            Пытается найти соответствие в базе голосов, если находит, то возвращает id спикера.
            Если голос не найден, то создается, сохраняется в базе и возвращается новый спикер.
        """
        # Расчет эмбеддинга спикера и поиск спикера по центроидам эмбеддингов
        # 1. Обрезка правого края с тишиной для коротких фраз
        if len(seg) <= int(1.5 * SR):
            vad_seg = segment_utils.trim_silence_fix_end(seg)
            # segment_utils.visualize_segment_energy(vad_seg)
        else:
            vad_seg = seg

        # 2. Логика определения спикера (только для качественных сегментов)
        if len(vad_seg) >= int(MIN_SEARCH_SEG_LEN * SR):
            emb = self._normalize_vector(compute_embedding(self._extractor, vad_seg))
            resolve_result = self._search_or_create_speaker(vad_seg, emb)
            if resolve_result.speaker is not None:
                resolve_result.speaker.session_count += 1
                resolve_result.speaker.session_time += len(seg) / SR
                resolve_result.speaker.total_count += 1
                resolve_result.speaker.total_time += len(seg) / SR
            else:
                raise ValueError("Объект resolve_result.speaker не дожен быть None")
        else:
            # Сегмент короткий: создаем результат без спикера
            resolve_result = ResolveResult(
                speaker = None,
                cos_similarity = -1,
            )

        return resolve_result

def get_speaker_resolver(pl_conf: PipelineConfig) -> SpeakerResolver:
    """Создает резольвер нужно типа, в соответствии с конфигурацией пайплайна"""
    if pl_conf.diar_vad.use_oracle:
        return OracleSpeakerResolver(pl_conf = pl_conf)
    if pl_conf.diar_vad.resolving_mode is SpeakerResolvingMode.NONE:
        return NoneSpeakerResolver()
    if pl_conf.diar_vad.resolving_mode is SpeakerResolvingMode.VAD_SIMPLE_CENTROID:
        return VadSpeakerResolver(pl_conf = pl_conf)

    raise ValueError(f"Тип диаризации {pl_conf.diar_vad.resolving_mode} пока не поддерживается")
