"""Модуль с раннером бенчмарков"""
# Запуск: PYTHONPATH=src python src/benchmark/benchmark_runner.py
from dataclasses import dataclass
from pathlib import Path
import time
from config import PipelineConfig, config
from pipeline_vad import AsrPipeline
from entities import PipelineResult, AudioSegment
import common_utils
from benchmark.markup_storage import load_scenario_from_yaml, load_markup_from_yaml

@dataclass
class BenchmarkSpec:
    """
    Атомарная спецификация для тестирования одного аудиофайла.
    Полностью описывает КТО, ЧТО и КАК обрабатывает.
    """
    spec_id: str        # Уникальный ID задачи (например, "sample_001__all_oracle")
    # Данные
    audio_path: Path
    ground_truth_path: Path | None = None       # Путь к эталонной разметке (JSON/RTTM)
    # Символические имена моделей
    asr_model_name: str | None = None           # Например: "whisper-large-v3"
    embedding_model_name: str | None = None     # Например: "pyannote-wespeaker"
    vad_model_name: str = "silero"              # По умолчанию Silero
    # Матрица включения Оракулов
    use_oracle_vad: bool = False
    use_oracle_asr: bool = False
    use_oracle_diarization: bool = False
    # Профиль настроек гиперпараметров
    profile: str | None = None              # Например: meeting или noisy_environment
    # Метаданные для аналитики метрик
    metadata: dict[str, any] | None = None  # {"dataset_name": "voxceleb", "snr_level": "low"}

class BenchmarkRunner:
    """Класс обеспечивает запуск бенчмарков"""
    def __init__(self, benchmark_specs: list[BenchmarkSpec]):
        self.benchmark_specs = benchmark_specs

    def load_ground_truth(self, gt_file_path: Path) -> list[AudioSegment]:
        """Загружает эталонную разметку по пути к файлу с разметкой"""
        if "speaker00" in str(gt_file_path) and ".opus.yaml" in str(gt_file_path):
            # Если в имени файла есть фрагменты имени исходных аудиофайлов
            # То загружаем эталонную разметку из рзметки исходного аудиофайла
            _, audio_file = load_markup_from_yaml(gt_file_path)
            gt_markup = audio_file.segments
            # Создаем новый список без объектов с phrase_id == 0
            gt_markup = [seg for seg in gt_markup if seg.phrase_id != 0]
        else:
            gt_markup: list[AudioSegment] = []
            # Загружаем эталонную размету из сценария
            gt_scenario = load_scenario_from_yaml(gt_file_path)
            # Конвертируем эталонную разметку в список AudioSegment
            for gt_episode in gt_scenario.episodes:
                for gt_event in gt_episode.events:
                    gt_markup.append(
                        AudioSegment(
                            start_time = gt_event.start,
                            end_time = gt_event.end,
                            text = gt_event.text,
                        )
                    )
        return gt_markup

    def build_pl_config(self, bm_spec: BenchmarkSpec | None = None) -> PipelineConfig:
        """Создает конфиг пайплайна на основе спецификации бенчмарка"""
        pl_config = PipelineConfig(config = config)
        if bm_spec:
            pl_config.use_oracle_vad = bm_spec.use_oracle_vad

        return pl_config

    def run_single_combination(self) -> list[PipelineResult]:
        """Запуск одной конкретной конфигурации на всем датасете."""
        results: list[PipelineResult] = []
        print(f"Начинаем перебор и запуск всех benchmark_specs, {len(self.benchmark_specs)} штук")
        for bm_spec in self.benchmark_specs:
            print(f"Начинаем бенчмарк: {bm_spec.spec_id}")
            audio_path = bm_spec.audio_path
            print(f"Путь к аудио: {audio_path}")

            # Создаем пайплайны под конкретный аудиофайл с его GT (для оракулов)
            pl_config = self.build_pl_config(bm_spec = bm_spec)
            pl = AsrPipeline(pl_config = pl_config)

            # Если включен режим Оракула, то загружаем эталонную разметку
            if (
                bm_spec.use_oracle_vad
                or bm_spec.use_oracle_asr
                or bm_spec.use_oracle_diarization
            ):
                gt_markup = self.load_ground_truth(bm_spec.ground_truth_path)
            else:
                gt_markup = None

            # Засекаем время запуска пайплайна
            pl_start_time = time.perf_counter()

            # Выполняем инференс
            # pipeline_result = pl.run(audio_path)
            print("Запускаем пайплайн...")
            for seg in pl.run_as_stream(audio_path = audio_path, markup_segments = gt_markup):
                ts_start = common_utils.format_timestamp(seg.start_time)
                ts_end = common_utils.format_timestamp(seg.end_time)
                print(f"[{ts_start}-{ts_end}] {seg.text}")
            pipeline_result = pl.pipeline_result

            # Засекаем время окончания распознавания
            pl_end_time = time.perf_counter()
            pipeline_result.run_time = pl_end_time - pl_start_time

            print(f"Время распознавания: {pipeline_result.run_time:.6f} секунд")

            results.append(pipeline_result)
        return results

def generate_benchmark_suite(audio_path: Path, gt_file: Path) -> list[BenchmarkSpec]:
    """Генерирует матрицу тестов для одного файла."""
    # Создаем комбинации: только реальный пайплайн и полностью оракульный
    return [
        BenchmarkSpec(
            spec_id = f"{audio_path.stem}_oracle_vad_only",
            audio_path = audio_path,
            ground_truth_path = gt_file,
        )
    ]

def main():
    """Точка входа для тестирования в ходе разработки"""
    # Оркестратор просто итерируется по плоскому списку спецификаций
    # bm_spesc = generate_benchmark_suite(
    #     audio_path = Path("dataset/scenario_recipe_1spk_monologue.opus"),
    #     gt_file = Path("dataset/scenario_recipe_1spk_monologue.yaml")
    #     )
    # bm_spesc = generate_benchmark_suite(
    #     audio_path = Path("dataset/speaker001.opus"),
    #     gt_file = Path("dataset/speaker001.opus.yaml")
    #     )
    bm_spesc = generate_benchmark_suite(
        audio_path = Path("dataset/speaker002.opus"),
        gt_file = Path("dataset/speaker002.opus.yaml")
        )
    bm_spesc[0].use_oracle_vad = True
    # Отправляем specs_pool в пул потоков/процессов на выполнение...
    bm_runner = BenchmarkRunner(bm_spesc)
    bm_runner.run_single_combination()

if __name__ == "__main__":
    main()
