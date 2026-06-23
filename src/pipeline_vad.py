"""Пайплан для распознавания и диаризации при помощи VAD и центроидов"""
import sys
from collections.abc import Generator
import numpy as np
from config import PipelineConfig, SR
from entities import PipelineResult
import ffmpeg_utils
import model_utils
from speaker_storage import Speaker, AudioFile, AudioSegment
from diarization_utils import SpeakerResolver, SpeakerResolvingMode
import vad_utils
import asr_utils

class BaseVadPipeline:
    """Базовый пайплайн для распознавания и диаризации при помощи VAD"""
    def __init__(self, pl_config: PipelineConfig, speakers: list[Speaker] | None = None):
        self._pl_config = pl_config
        self._config = pl_config.config
        self._speakers = speakers
        self.pipeline_result: PipelineResult | None = None
        self._speaker_resolver: SpeakerResolver | None = None
        # Список сегментов разметки для режимов OracleVAD, OracleASR, OracleDiarization
        self.markup_segments: list[AudioSegment] | None = None
        self._init_models()

    def _init_models(self):
        """Инициализация моделей"""
        # Инициализируем VAD
        if self._pl_config.is_oracle_vad:
            # Создаем OracleVAD
            self._vad = vad_utils.OracleVAD(buffer_size_in_seconds = 100, padding_seconds = 0.0)
            self._window_size = 512
        else:
            # Создаем оригинальный VAD ()
            self._vad, self._window_size = model_utils.load_vad(
                vad_model = self._config.get_vad_model()['model'],
                threshold = self._pl_config.vad_threshold,
                min_silence = self._pl_config.vad_min_silence,
                min_speech = self._pl_config.vad_min_speech,
                max_speech = self._pl_config.vad_max_speech,
            )

        # Инициализируем ASR распознаватель
        self._recognizer = model_utils.load_asr(
            num_threads = self._pl_config.num_threads,
            provider = self._pl_config.provider
        )

    def set_markup_segments(self, markup_segments: list[AudioSegment] | None = None):
        """Устанавливает эталонную разметку во всех Оракулах пайплайна, которые включены"""
        # Если пайплайн в режиме OracleVAD, устанавливаем эталонную разметку
        if self._pl_config.is_oracle_vad:
            self.markup_segments = markup_segments
            self._vad.set_markup_segments(markup_segments)

    def run_as_stream(
        self, audio_path: str = "mic",
        markup_segments: list[AudioSegment] | None = None,
    ) -> Generator[AudioSegment, None, None]:
        """
        Запуск пайплайна для аудио с источником в audio_path
        и эталонной разметкой в markup_segments для режима Оракула
        """
        self.pipeline_result = None
        # Вызываем установку эталонной разметки для Оракулов, которые включены
        self.set_markup_segments(markup_segments)
        self._vad.reset() # Сброс внутреннего состояния VAD в исходное

        if audio_path == "mic":
            proc = ffmpeg_utils.make_ffmpeg_proc_for_pulse_default()
        else:
            proc = ffmpeg_utils.make_ffmpeg_proc_for_file(audio_path)
        if proc.stdout is None:
            print("ffmpeg stdout is None", file=sys.stderr)
            ffmpeg_utils.close_ffmpeg_proc(proc)
            sys.exit(1)

        try:
            # Оснвной цикл
            segments: list[AudioSegment] = []
            audio_file = AudioFile(file_path = audio_path, segments = segments)
            while True:
                # Пробуем прочитать полный блок (window_size == 512) данных (0.032 секунды аудио)
                samples = ffmpeg_utils.read_samples(proc, self._window_size)
                # Если блок пустой или неполный, то игнорируем его и переходим к выталкиванию
                # из VAD тишиной последней незавершенной фразы, если она есть
                if len(samples) == 0 or len(samples) < self._window_size:
                    break

                # Передает в VAD очередной блок данных. Когда VAD определяет начало фразы, он
                # начинает накапливать фрагменты фразы до тех пор, пока не определит завершение
                # фразы.
                # После этого vad.empty() возвращает False и фразу можно забирать целиком.
                # После извлечения фразы из VAD методом vad.pop(), VAD становится пустым,
                # и vad.empty() == True
                self._vad.accept_waveform(samples)
                for segment, t_start, t_end in vad_utils.get_speec_segments(self._vad):
                    # Распознаем (ASR) полученный из VAD сегмент
                    text = asr_utils.decode_asr(self._recognizer, segment)
                    # Распознаем спикера
                    resolve_result = self._speaker_resolver.resolve(segment, t_start, t_end)

                    if text:
                        segment = AudioSegment(
                            audio_file = audio_file,
                            speaker = resolve_result.speaker,
                            cos_similarity = resolve_result.cos_similarity,
                            start_time = t_start,
                            end_time = t_end,
                            text = text,
                        )
                        segments.append(segment)
                        yield segment

            # Проталкиваем в VAD последнюю неоконченную фразу 1 секундой тишины (нулевые данные)
            zeros = np.zeros(self._window_size, dtype=np.float32)
            for _ in range(int(SR / self._window_size) + 2):
                self._vad.accept_waveform(zeros)
                for segment, t_start, t_end in vad_utils.get_speec_segments(self._vad):
                    # Распознаем (ASR) полученный из VAD сегмент
                    text = asr_utils.decode_asr(self._recognizer, segment)
                    # Распознаем спикера
                    resolve_result = self._speaker_resolver.resolve(segment, t_start, t_end)

                    if text:
                        segment = AudioSegment(
                            audio_file = audio_file,
                            speaker = resolve_result.speaker,
                            cos_similarity = resolve_result.cos_similarity,
                            start_time = t_start,
                            end_time = t_end,
                            text = text,
                        )
                        segments.append(segment)
                        yield segment
        finally:
            ffmpeg_utils.close_ffmpeg_proc(proc)

        self.pipeline_result = PipelineResult(
            pipeline_type = None,
            speakers = self._speakers,
            file = audio_file,
            segments = segments,
        )

    def run(
        self, audio_path: str = "mic",
        markup_segments: list[AudioSegment] | None = None,
    ) -> PipelineResult:
        """Метод сразу возвращает конечный результат"""
        # Истощает собственный генератор и возвращает готовый результат
        for _ in self.run_as_stream(audio_path = audio_path, markup_segments = markup_segments):
            pass

        return self.pipeline_result


class AsrPipeline(BaseVadPipeline):
    """Пайплайн для распознавания при помощи VAD"""
    def _init_models(self):
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса в холостом режиме SpeakerResolvingMode.NONE
        self._speaker_resolver = SpeakerResolver(
            num_threads = self._pl_config.num_threads,
            spk_threshold = self._pl_config.spk_threshold,
            resolving_mode = SpeakerResolvingMode.NONE,
            speakers = self._speakers,
        )

class ManagerDiarizationPipeline(BaseVadPipeline):
    """Пайплайн для распознавания и диаризации при помощи VAD и менеджера спикеров"""
    def _init_models(self):
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса
        self._speaker_resolver = SpeakerResolver(
            num_threads = self._pl_config.num_threads,
            spk_threshold = self._pl_config.spk_threshold,
            resolving_mode = SpeakerResolvingMode.VAD_SPEAKER_MANAGER,
            speakers = self._speakers,
        )

class CentroidDiarizationPipeline(BaseVadPipeline):
    """Пайплайн для распознавания и диаризации при помощи VAD и центроидов"""
    def _init_models(self):
        """Инициализация моделей"""
        # Запуск инициализации моделей в родительском классе
        super()._init_models()

        #Инициализируем распознаватель голоса
        self._speaker_resolver = SpeakerResolver(
            num_threads = self._pl_config.num_threads,
            spk_threshold = self._pl_config.spk_threshold,
            resolving_mode = SpeakerResolvingMode.VAD_SIMPLE_CENTROID,
            speakers = self._speakers,
        )
