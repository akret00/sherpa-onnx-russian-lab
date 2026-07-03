"""
Модуль рассчитывает метрики распознавания речи:
- WER (Word Error Rate) - Показывает общую точность. Помогает сразу отсеять слабые модели.
- Показатели S (Замены), D (Пропуски), I (Вставки) в абсолютных числах. Это ключ к настройке
    пайплайна (VAD — детектора тишины, аудио-фильтров, параметров декодера).
    - Если много Пропусков (D) — ваш VAD слишком агрессивно режет тишину и съедает окончания
        слов, либо модель «глотает» быструю речь. Нужно снижать порог чувствительности VAD.
    - Если много Замен (S) — модель плохо различает похожие звуки. Здесь поможет смена
        акустической модели, добавление языковой модели (Language Model) или замена промпта
        (для моделей типа Whisper).
- CER (Character Error Rate) — Коэффициент ошибок в символах. Помогает понять, насколько
    сильно ошибается модель. Если модель вместо «корова» распознала «карова» — для WER
    это стопроцентная ошибка (целое слово неверно). Но для человека, это мелкая опечатка.
    Если WER высокий, а CER низкий — модель ошибается лишь в отдельных буквах
    (окончаниях, падежах), и ее можно брать в работу.
"""
# Запуск: PYTHONPATH=src python src/benchmark/scorer_wer.py
import jiwer
from config import PipelineType, BASE_DIR
from benchmark.experiment_runner import ExperimentRunner
from benchmark.experiment_entities import (
    ExperimentSpec, PipelineResultExperiment,
    MetricExpWER, MetricWER, MetricCER,
)
from benchmark.experiment_storage import (
    load_plres_exp_from_yaml, export_plres_exp_to_yaml,
    export_metrics_wer_to_yaml,
    EXP_RUNS_BASE_DIR
)

class CustomNormalisationTransform(jiwer.AbstractTransform):
    """Кастомный класс для нормализации."""
    def process_list(self, inp: list[str]) -> list[str]:
        # Логика обработки списка строк
        return [self.process_string(text) for text in inp]

    def process_string(self, s: str) -> str:
        # Логика обработки одной строки
        norm_text = s
        if "о'кей" in norm_text:
            norm_text = norm_text.replace("о'кей", "окей")
        if "кгц" in norm_text:
            norm_text = norm_text.replace("кгц", "килогерц")
        if "19:45" in norm_text:
            norm_text = norm_text.replace("19:45", "девятнадцать сорок пять")
        if "3 450" in norm_text:
            norm_text = norm_text.replace("3 450", "три тысячи четыреста пятьдесят")
        if "7701" in norm_text:
            norm_text = norm_text.replace("7701", "семьдесят семь ноль один")
        if "2024" in norm_text:
            norm_text = norm_text.replace("2024", "два ноль два четыре")
        if "500" in norm_text:
            norm_text = norm_text.replace("500", "пятьсот")
        if "101" in norm_text:
            norm_text = norm_text.replace("101", "сто один")
        if "102" in norm_text:
            norm_text = norm_text.replace("102", "сто два")
        if "103" in norm_text:
            norm_text = norm_text.replace("103", "сто три")
        if "001" in norm_text:
            norm_text = norm_text.replace("001", " ноль ноль один")
        if "16" in norm_text:
            norm_text = norm_text.replace("16", "шестнадцать")
        if "12" in norm_text:
            norm_text = norm_text.replace("12", "двенадцать")
        if "10" in norm_text:
            norm_text = norm_text.replace("10", "десять")
        if "2" in norm_text:
            norm_text = norm_text.replace("2", "два")
        if "7" in norm_text:
            norm_text = norm_text.replace("7", "семь")
        if "9" in norm_text:
            norm_text = norm_text.replace("9", "девять")
        if "₽" in norm_text:
            norm_text = norm_text.replace("₽", "рублей")
        if "%" in norm_text:
            norm_text = norm_text.replace("%", " процентов")

        return norm_text

# Создаем базовый нормализатор
base_transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    CustomNormalisationTransform(),
    jiwer.SubstituteRegexes({r"[^\w\s]": " "}), # Заменяет любой знак пунктуации на пробел
    # jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def calc_wer(
    clean_ref: str | list[str],
    clean_hyp: str | list[str],
    obj_id: str | None = None,
    use_align: bool = False
) -> MetricWER:
    """Считает метрику WER"""
    output_wer = jiwer.process_words(reference = clean_ref, hypothesis = clean_hyp)
    alignment = None
    if use_align:
        alignment = jiwer.visualize_alignment(output = output_wer, show_measures = False)

    return MetricWER(
        obj_id = obj_id,
        wer = output_wer.wer,
        gt_words_count = output_wer.hits + output_wer.substitutions + output_wer.deletions,
        substitutions = output_wer.substitutions,
        deletions = output_wer.deletions,
        insertions = output_wer.deletions,
        alignment = alignment,
    )

def calc_cer(
    clean_ref: str | list[str],
    clean_hyp: str | list[str],
    obj_id: str | None = None,
    use_align: bool = False
) -> MetricCER:
    """Считает метрику CER"""
    output_cer = jiwer.process_characters(reference = clean_ref, hypothesis = clean_hyp)
    alignment = None
    if use_align:
        alignment = jiwer.visualize_alignment(output = output_cer, show_measures = False)

    return MetricCER(
        obj_id = obj_id,
        cer = output_cer.cer,
        gt_chars_count = output_cer.hits + output_cer.substitutions + output_cer.deletions,
        substitutions = output_cer.substitutions,
        deletions = output_cer.deletions,
        insertions = output_cer.deletions,
        alignment = alignment,
    )

def calc_wer_total(pl_res_exp: PipelineResultExperiment) -> MetricExpWER:
    """Рассчитывает общие и построчные метрики WER и CER для pl_res_exp или exp_id"""
    references: list[str] = []
    hypothesis: list[str] = []

    # Готовим списки эталоных фраз и гипотез
    for seg in pl_res_exp.markup_segments:
        references.append(seg.text)
    for seg in pl_res_exp.segments:
        hypothesis.append(seg.text)

    clean_ref = base_transformation(references)
    clean_hyp = base_transformation(hypothesis)

    # Рассчет общих метрик
    exp_wer = calc_wer(
        clean_ref = clean_ref,
        clean_hyp = clean_hyp,
        obj_id = pl_res_exp.exp_id,
        use_align = False
    )

    exp_cer = calc_cer(
        clean_ref = clean_ref,
        clean_hyp = clean_hyp,
        obj_id = pl_res_exp.exp_id,
        use_align = False
    )

    # Расчет метрик для фраз с ошибками
    err_segments_wer: list[MetricWER] = []
    err_segments_cer: list[MetricCER] = []

    for idx, (ref, hyp) in enumerate(zip(clean_ref, clean_hyp)):
        seg_wer = calc_wer(
            clean_ref = ref,
            clean_hyp = hyp,
            obj_id = str(idx),
            use_align = True
        )

        seg_cer = calc_cer(
            clean_ref = ref,
            clean_hyp = hyp,
            obj_id = str(idx),
            use_align = True
        )

        # Выявление фраз с ошибками
        if seg_wer.substitutions + seg_wer.deletions + seg_wer.insertions > 0:
            err_segments_wer.append(seg_wer)

        if seg_cer.substitutions + seg_cer.deletions + seg_cer.insertions > 0:
            err_segments_cer.append(seg_cer)

    return MetricExpWER(
        obj_id = pl_res_exp.exp_id,
        seg_count = len(pl_res_exp.segments),
        exp_wer = exp_wer,
        exp_cer = exp_cer,
        err_segments_wer = err_segments_wer,
        err_segments_cer = err_segments_cer,
    )


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
        total_wer = calc_wer_total(pl_res_exp = pl_result)
    else: # Сохраненного результата нет, запускаем бенчмарк и сохраняем результаты
        print("Сохраненные результаты отсутствуют, запускается бенчмарк...")
        audio_path = BASE_DIR / "dataset" / "speaker001.opus"
        gt_file = BASE_DIR / "dataset" / "speaker001.opus.yaml"
        exp_specs = [
            ExperimentSpec(
                spec_id = f"{audio_path.stem}_oracle_vad_only",
                audio_path = str(audio_path),
                ground_truth_path = str(gt_file),
                use_oracle_vad = True,
                pipeline_type = PipelineType.ASR_PIPELINE,
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
        total_wer = calc_wer_total(pl_res_exp = pl_result)

    # Сохраняем рассчитанные метрики
    export_metrics_wer_to_yaml(metrics_exp_wer = total_wer, exp_id = exp_id)

    print(f"Количество ref: {len(pl_result.markup_segments)} hyp: {len(pl_result.segments)}")

    print("========== Детализация ошибок по фразам ==========")
    for idx, seg in enumerate(total_wer.err_segments_wer):
        print(
            f"[Фраза: {idx}] Найдено ошибок: {seg.deletions + seg.insertions + seg.substitutions}"
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
    print(f"Всего строк с ошибками: {len(total_wer.err_segments_wer)} из {total_wer.seg_count}")

    print(f"Общий CER корпуса: {total_wer.exp_cer.cer:.4f}")
    print(f"Всего символов в эталоне: {total_wer.exp_cer.gt_chars_count}")
    print(f"Всего замен (S): {total_wer.exp_cer.substitutions}")
    print(f"Всего удалений (D): {total_wer.exp_cer.deletions}")
    print(f"Всего вставок (I): {total_wer.exp_cer.insertions}")

if __name__ == "__main__":
    main()
