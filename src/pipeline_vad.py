"""Пайплан для распознавания и диаризации при помощи VAD и центроидов"""
import time
from datetime import datetime
from collections.abc import Generator
import numpy as np
from config import PipelineConfig, SpeakerResolvingMode, SR
import ffmpeg_utils
from entities import AudioFile, AudioSegment, PipelineResult
from diarization_utils import SpeakerResolver
import vad_utils
import asr_utils
from common_utils import get_package_version

class BaseVadPipeline:
    """Базовый пайплайн для распознавания и диаризации при помощи VAD"""
    def __init__(self, pl_config: PipelineConfig):
        self._pl_config = pl_config
        self._speaker_resolver: SpeakerResolver
        self.pipeline_result: PipelineResult | None = None
        # Список сегментов разметки для режимов OracleVAD, OracleASR, OracleDiarization
        self.markup_segments: list[AudioSegment] | None = None
        self.test_duration_sec: float = 0.0
        self._init_models()

    def _init_models(self) -> None:
        """Инициализация моделей"""
        # Инициализируем VAD
        self._vad: vad_utils.BaseVAD
        if self._pl_config.vad.use_oracle:
            # Создаем OracleVAD
            self._vad = vad_utils.OracleVAD(buffer_size_in_seconds = 100, padding_seconds = 0.0)
            self._window_size = 512
        else:
            # Создаем оригинальный VAD ()
            self._vad = vad_utils.SherpaVADAdapter(
                pl_conf = self._pl_config,
                buffer_size_in_seconds = self._pl_config.vad.max_speech * 2 # *2 на всякий случай
            )
            self._window_size = self._vad.window_size

        # Инициализируем ASR распознаватель
        self._asr: asr_utils.BaseASR
        if self._pl_config.asr.use_oracle:
            # Создаем OracleASR
            self._asr = asr_utils.OracleASR()
        else:
            # Создаем оригинальный ASR распознаватель
            self._asr = asr_utils.SherpaASRAdapter(pl_config = self._pl_config)

    def set_markup_segments(self, markup_segments: list[AudioSegment] | None = None) -> None:
        """Устанавливает эталонную разметку во всех Оракулах пайплайна, которые включены"""
        # Если markup_segments не задан, то пропускаем установку
        if markup_segments is None:
            self.markup_segments = None
            self.test_duration_sec = 0.0
            return
        # Проверка на отрицательную или нулевую длительность
        for i, seg in enumerate(markup_segments):
            if seg.start_time >= seg.end_time:
                raise ValueError(
                    f"Сегмент {i}: start_time ({seg.start_time}) >= end_time ({seg.end_time})"
                )

        # Сортируем сегменты по start_time в порядке возрастания
        self.markup_segments = sorted(markup_segments, key=lambda x: x.start_time)
        # Устанавливаем длительность пустого тестового аудио
        self.test_duration_sec = self.markup_segments[-1].end_time

        # Если пайплайн в режиме OracleVAD, устанавливаем эталонную разметку
        if self._pl_config.vad.use_oracle:
            if isinstance(self._vad, vad_utils.OracleVAD):
                self._vad.set_markup_segments(self.markup_segments)
            else:
                raise ValueError(f"Недопустимый тип класса vad: {self._vad.__class__.__name__}")
        # Если пайплайн в режиме OracleASR, устанавливаем эталонную разметку
        if self._pl_config.asr.use_oracle:
            if isinstance(self._asr, asr_utils.OracleASR):
                self._asr.set_markup_segments(self.markup_segments)
            else:
                raise ValueError("В режиме Оракула атрибут _asr должен иметь тип OracleASR")
        if self._pl_config.diar_vad.use_oracle:
            if isinstance(self._speaker_resolver, SpeakerResolver):
                self._speaker_resolver.set_markup_segments(self.markup_segments)

    def run_as_stream(
        self, audio_path: str = "mic",
        markup_segments: list[AudioSegment] | None = None,
    ) -> Generator[AudioSegment, None, None]:
        """
        Запуск пайплайна для аудио с источником в audio_path
        и эталонной разметкой в markup_segments для режима Оракула
        """
        # Засекаем время запуска пайплайна
        pl_start_time = time.perf_counter()
        pl_start_datetime = datetime.now()

        self.pipeline_result = None
        # Вызываем установку эталонной разметки для Оракулов, которые включены
        self.set_markup_segments(markup_segments)
        # Сброс внутреннего состояния Оракула VAD в исходное
        if isinstance(self._vad, vad_utils.OracleVAD):
            self._vad.reset()
        # Сброс внутреннего состояния Оракула ASR в исходное
        if isinstance(self._asr, asr_utils.OracleASR):
            self._asr.reset()
        # Сброс состояния Оракула резольвера спикеров в исходное состояние
        if isinstance(self._speaker_resolver, SpeakerResolver):
            self._speaker_resolver.reset()

        with ffmpeg_utils.AudioStreamReader(
            path = audio_path, duration_sec = self.test_duration_sec
        ) as as_reader:
            segments: list[AudioSegment] = []
            audio_file = AudioFile(file_path = audio_path, segments = segments)
            # Основной цикл
            for samples in as_reader.iter_chunks():
                # Передает в VAD очередной блок данных. Когда VAD определяет начало фразы, он
                # начинает накапливать фрагменты фразы до тех пор, пока не определит завершение
                # фразы.
                # После этого vad.empty() возвращает False и фразу можно забирать целиком.
                # После извлечения фразы из VAD методом vad.pop(), VAD становится пустым,
                # и vad.empty() == True
                self._vad.accept_waveform(samples)
                if not self._vad.empty():
                    for vad_seg, t_start, t_end in vad_utils.get_speec_segments(self._vad):
                        # Распознаем (ASR) полученный из VAD сегмент
                        text = self._asr.decode_asr(samples_f32 = vad_seg, t_start = t_start)
                        # Распознаем спикера
                        resolve_result = self._speaker_resolver.resolve(vad_seg, t_start, t_end)

                        # if text:
                        segment = AudioSegment(
                            # audio_file = audio_file,
                            speaker_id = (
                                resolve_result.speaker.id if resolve_result.speaker is not None
                                else None
                            ),
                            speaker = resolve_result.speaker,
                            cos_similarity = resolve_result.cos_similarity,
                            start_time = t_start,
                            end_time = t_end,
                            text = text,
                        )
                        segments.append(segment)
                        yield segment

            # Проталкиваем в VAD последнюю неоконченную фразу 2 секундами тишины (нулевые данные)
            zeros = np.zeros(self._window_size, dtype=np.float32)
            for _ in range(int(SR / self._window_size) * 2):
                self._vad.accept_waveform(zeros)
                for vad_seg, t_start, t_end in vad_utils.get_speec_segments(self._vad):
                    # Распознаем (ASR) полученный из VAD сегмент
                    text = self._asr.decode_asr(samples_f32 = vad_seg, t_start = t_start)
                    # Распознаем спикера
                    resolve_result = self._speaker_resolver.resolve(vad_seg, t_start, t_end)

                    if text:
                        segment = AudioSegment(
                            # audio_file = audio_file,
                            speaker_id = (
                                resolve_result.speaker.id if resolve_result.speaker is not None
                                else None
                            ),
                            speaker = resolve_result.speaker,
                            cos_similarity = resolve_result.cos_similarity,
                            start_time = t_start,
                            end_time = t_end,
                            text = text,
                        )
                        segments.append(segment)
                        yield segment

        # Сохраняем обновленные данны спикеров (в т.ч. эмбединги и счетчики) в хранилище
        self._speaker_resolver.save_all_speakers()

        # Засекаем время окончания работы пайплайна
        pl_end_time = time.perf_counter()

        # Формируем результат работы пайплайна
        self.pipeline_result = PipelineResult(
            pl_config = self._pl_config,
            speakers = self._speaker_resolver.get_speakers(),
            file = audio_file,
            segments = segments,
            markup_segments = self.markup_segments,
            start_time = pl_start_datetime,
            proc_time = pl_end_time - pl_start_time,
            total_ram = None,   # Расчет пикового потребления ОЗУ отложен на потом
            sherpa_version = get_package_version("sherpa_onnx"),
        )

    def run(
        self, audio_path: str = "mic",
        markup_segments: list[AudioSegment] | None = None,
    ) -> PipelineResult:
        """Метод сразу возвращает конечный результат"""
        # Истощает собственный генератор и возвращает готовый результат
        for _ in self.run_as_stream(audio_path = audio_path, markup_segments = markup_segments):
            pass

        if self.pipeline_result is not None:
            return self.pipeline_result
        else:
            raise ValueError("self.pipeline_result не должен быть None")


class AsrPipeline(BaseVadPipeline):
    """Пайплайн для распознавания при помощи VAD"""
    def _init_models(self) -> None:
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса в холостом режиме SpeakerResolvingMode.NONE
        self._speaker_resolver = SpeakerResolver(pl_conf = self._pl_config)

class ManagerDiarizationPipeline(BaseVadPipeline):
    """Пайплайн для распознавания и диаризации при помощи VAD и менеджера спикеров"""
    def _init_models(self) -> None:
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса
        self._pl_config.diar_vad.resolving_mode = SpeakerResolvingMode.VAD_SPEAKER_MANAGER
        self._speaker_resolver = SpeakerResolver(pl_conf = self._pl_config)

class CentroidDiarizationPipeline(BaseVadPipeline):
    """Пайплайн для распознавания и диаризации при помощи VAD и центроидов"""
    def _init_models(self) -> None:
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса
        self._pl_config.diar_vad.resolving_mode = SpeakerResolvingMode.VAD_SIMPLE_CENTROID
        self._speaker_resolver = SpeakerResolver(pl_conf = self._pl_config)
