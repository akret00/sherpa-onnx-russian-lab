"""Модуль с раннером бенчмарков"""
# Запуск: PYTHONPATH=src python src/benchmark/experiment_runner.py
from dataclasses import asdict
from pathlib import Path
from config import PipelineType, PipelineConfig, config
from pipeline_vad import (
    AsrPipeline,
    ManagerDiarizationPipeline,
    CentroidDiarizationPipeline,
)
import common_utils
from benchmark.dataset_storage import (
    AudioSegmentMarkup,
    load_scenario_from_yaml,
    load_markup_from_yaml,
)
from benchmark.experiment_entities import ExperimentSpec, PipelineResultExperiment


class ExperimentRunner:
    """Класс обеспечивает запуск бенчмарков"""
    def __init__(self, experiment_specs: list[ExperimentSpec]):
        self.experiment_specs = experiment_specs

    def load_ground_truth(self, gt_file_path: Path) -> list[AudioSegmentMarkup]:
        """Загружает эталонную разметку по пути к файлу с разметкой"""
        if "speaker00" in str(gt_file_path) and ".opus.yaml" in str(gt_file_path):
            # Если в имени файла есть фрагменты имени исходных аудиофайлов
            # То загружаем эталонную разметку из рзметки исходного аудиофайла
            _, audio_file = load_markup_from_yaml(gt_file_path)
            gt_markup = audio_file.segments
            # Создаем новый список без объектов с phrase_id == 0
            gt_markup = [seg for seg in gt_markup if seg.phrase_id != 0]
        else:
            gt_markup: list[AudioSegmentMarkup] = []
            # Загружаем эталонную размету из сценария
            gt_scenario = load_scenario_from_yaml(gt_file_path)
            # Конвертируем эталонную разметку в список AudioSegment
            for gt_episode in gt_scenario.episodes:
                for gt_event in gt_episode.events:
                    gt_markup.append(
                        AudioSegmentMarkup(
                            start_time = gt_event.start,
                            end_time = gt_event.end,
                            text = gt_event.text,
                        )
                    )
        return gt_markup

    def build_pl_config(self, exp_spec: ExperimentSpec | None = None) -> PipelineConfig:
        """Создает конфиг пайплайна на основе спецификации бенчмарка"""
        if exp_spec:
            pl_config = config.get_new_pipeline_config(
                vad_model_name = exp_spec.vad_model_name,
                asr_model_name = exp_spec.asr_model_name,
                embed_model_name = exp_spec.embed_model_name,
                segmentation_model_name = exp_spec.segmentation_model_name,
            )
            pl_config.vad.use_oracle = exp_spec.use_oracle_vad
            pl_config.asr.use_oracle = exp_spec.use_oracle_asr
            pl_config.diar_vad.use_oracle = exp_spec.use_oracle_diarization
            pl_config.runtime.pipeline_type = exp_spec.pipeline_type
        else:
            pl_config = config.get_new_pipeline_config()

        return pl_config

    def get_result_exp_id(self, plres: PipelineResultExperiment) -> str:
        """Конструирует имя папки на основе данных из результатов эксперимента"""
        time = plres.start_time.strftime("%Y%m%d_%H%M%S")
        dataset = plres.exp_spec.dataset_view
        pl_type = plres.pl_config.runtime.pipeline_type.value
        vad = plres.pl_config.vad.model_short_name
        asr = plres.pl_config.asr.model_short_name
        embed = plres.pl_config.embed.model_short_name
        exp_id = f"{time}_{dataset}_{pl_type}_{vad}_{asr}_{embed}"
        return exp_id

    def run_single_combination(self) -> list[PipelineResultExperiment]:
        """Запуск одной конкретной конфигурации на всем датасете."""
        results: list[PipelineResultExperiment] = []
        print(f"Начинаем перебор и запуск всех experiment_specs, {len(self.experiment_specs)} штук")
        for exp_spec in self.experiment_specs:
            print(f"Начинаем бенчмарк: {exp_spec.spec_id}")
            audio_path = exp_spec.audio_path
            print(f"Путь к аудио: {audio_path}")

            # Создаем пайплайны под конкретный аудиофайл с его GT (для оракулов)
            pl_config = self.build_pl_config(exp_spec = exp_spec)
            if pl_config.runtime.pipeline_type is PipelineType.ASR_PIPELINE:
                pl = AsrPipeline(pl_config = pl_config)
            elif pl_config.runtime.pipeline_type is PipelineType.MANAGER_DIARIZ_PIPELINE:
                pl = ManagerDiarizationPipeline(pl_config = pl_config)
            elif pl_config.runtime.pipeline_type is PipelineType.CENTRIOD_DIARIZ_PIPELINE:
                pl = CentroidDiarizationPipeline(pl_config = pl_config)
            else:
                raise ValueError(f"Неизвестный тип пайплайна: {pl_config.runtime.pipeline_type}")

            # Если включен режим Оракула, то загружаем эталонную разметку
            if (
                exp_spec.use_oracle_vad
                or exp_spec.use_oracle_asr
                or exp_spec.use_oracle_diarization
            ):
                gt_markup = self.load_ground_truth(exp_spec.ground_truth_path)
            else:
                gt_markup = None

            # Выполняем инференс
            # pipeline_result = pl.run(audio_path)
            print("Запускаем пайплайн...")
            for seg in pl.run_as_stream(audio_path = audio_path, markup_segments = gt_markup):
                ts_start = common_utils.format_timestamp(seg.start_time)
                ts_end = common_utils.format_timestamp(seg.end_time)
                print(f"[{ts_start}-{ts_end}] {seg.text}")
            # Конвертирует результат пайплайна в PipelineResultExperiment
            pipeline_result_exp = PipelineResultExperiment(**pl.pipeline_result.__dict__)
            pipeline_result_exp.exp_spec = exp_spec
            pipeline_result_exp.exp_id = self.get_result_exp_id(plres = pipeline_result_exp)

            print(f"Время распознавания: {pipeline_result_exp.proc_time:.6f} секунд")

            results.append(pipeline_result_exp)
        return results

def main() -> None:
    """Точка входа для тестирования в ходе разработки"""
    # Оркестратор просто итерируется по плоскому списку спецификаций
    # audio_path = "dataset/scenario_recipe_1spk_monologue.opus"
    # gt_file = "dataset/scenario_recipe_1spk_monologue.yaml"
    audio_path = "dataset/speaker002.opus"
    gt_file = "dataset/speaker002.opus.yaml"
    exp_spec = [
        ExperimentSpec(
            spec_id = f"{Path(audio_path).stem}_oracle_vad_only",
            audio_path = audio_path,
            ground_truth_path = gt_file,
            use_oracle_vad = True,
            pipeline_type = PipelineType.ASR_PIPELINE,
        ),
    ]

    # Отправляем specs_pool в пул потоков/процессов на выполнение...
    exp_runner = ExperimentRunner(exp_spec)
    exp_runner.run_single_combination()

if __name__ == "__main__":
    main()
