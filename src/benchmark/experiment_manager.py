"""Модуль настраивает эксперименты, определяет типы метрик для них, считает и сохраняет их"""
# Запуск: PYTHONPATH=src python src/benchmark/experiment_manager.py

from pathlib import Path
from config import PipelineType, BASE_DIR, AUDIO_PATH_ORACLE_EMPTY
from benchmark.experiment_runner import ExperimentRunner
from benchmark.experiment_entities import (
    ExperimentSpec,
)
from benchmark.experiment_storage import (
    load_plres_exp_from_yaml, export_plres_exp_to_yaml,
    export_metrics_wer_to_yaml,
    EXP_RUNS_BASE_DIR
)
from benchmark.scorer_wer import calc_wer_total

def main() -> None:
    """Точка входа для тестирования"""
    exp_id = "20260703_150916_unknown_asr_sil_gig3pu8_3deres"

    # Проверяем наличие сохраненного результата эксперимента для отладки расчетов метрик
    runs_dir_path = EXP_RUNS_BASE_DIR / exp_id
    if runs_dir_path.exists(): # Сохраненные результаты есть
        pl_result = load_plres_exp_from_yaml(exp_id = exp_id)
        print(f"Загружены результаты эксперимента: {exp_id}")
        if pl_result is None:
            raise ValueError(f"Не найдены результаты эксперимента: {exp_id}")
        if pl_result.exp_spec.use_wer:
            total_wer = calc_wer_total(pl_res_exp = pl_result)
    else: # Сохраненного результата нет, запускаем бенчмарк и сохраняем результаты
        print("Сохраненные результаты отсутствуют, запускается бенчмарк...")
        audio_path = BASE_DIR / "dataset" / "speaker001.opus"
        # audio_path = Path(AUDIO_PATH_ORACLE_EMPTY)
        gt_file = BASE_DIR / "dataset" / "speaker001.opus.yaml"
        exp_specs = [
            ExperimentSpec(
                spec_id = f"{audio_path.stem}_oracle_vad_only",
                audio_path = str(audio_path),
                ground_truth_path = str(gt_file),
                # use_oracle_vad = True,
                # use_oracle_asr = True,
                # use_oracle_diarization = True,
                # pipeline_type = PipelineType.ASR_PIPELINE,
                pipeline_type = PipelineType.CENTRIOD_DIARIZ_PIPELINE,
                use_wer = True,
            ),
        ]

        # Отправляем exp_specs в очередь на выполнение...
        exp_runner = ExperimentRunner(exp_specs)
        pl_result = exp_runner.run_single_combination()[0]
        exp_id = pl_result.exp_id

        # Сохраняем результат в папку runs
        export_plres_exp_to_yaml(plres = pl_result)
        print(f"Кэш сохранен в папке: {EXP_RUNS_BASE_DIR / exp_id}")

        # Запускаем расчет метрик
        if pl_result.exp_spec.use_wer:
            total_wer = calc_wer_total(pl_res_exp = pl_result)

    if pl_result.exp_spec.use_wer:
        # Сохраняем рассчитанные метрики
        export_metrics_wer_to_yaml(metrics_exp_wer = total_wer, exp_id = exp_id)

        print(f"Количество ref: {len(pl_result.markup_segments)} hyp: {len(pl_result.segments)}")

        if total_wer.err_segments_wer is not None:
            print("========== Детализация ошибок по фразам ==========")
            for idx, seg in enumerate(total_wer.err_segments_wer):
                err_count = seg.deletions + seg.insertions + seg.substitutions
                print(
                    f"[Фраза: {idx}] Найдено ошибок: {err_count}"
                )
                # Печатаем красивое выравнивание
                print(seg.alignment)
                print("-" * 40)

        print("\n--- ИТОГОВАЯ СТАТИСТИКА КОРПУСА ---")
        print(f"Общий WER корпуса: {total_wer.exp_wer.wer:.4f}")
        print(f"Всего слов в эталоне: {total_wer.exp_wer.gt_words_count}")
        print(f"Всего замен (S): {total_wer.exp_wer.substitutions}")
        print(f"Всего удалений (D): {total_wer.exp_wer.deletions}")
        print(f"Всего вставок (I): {total_wer.exp_wer.insertions}")
        if total_wer.err_segments_wer:
            print(
                f"Всего строк с ошибками: {len(total_wer.err_segments_wer)} "
                f"из {total_wer.seg_count}"
            )

        print(f"Общий CER корпуса: {total_wer.exp_cer.cer:.4f}")
        print(f"Всего символов в эталоне: {total_wer.exp_cer.gt_chars_count}")
        print(f"Всего замен (S): {total_wer.exp_cer.substitutions}")
        print(f"Всего удалений (D): {total_wer.exp_cer.deletions}")
        print(f"Всего вставок (I): {total_wer.exp_cer.insertions}")

if __name__ == "__main__":
    main()
