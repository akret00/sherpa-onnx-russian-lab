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
from pathlib import Path
import jiwer
from benchmark.experiment_runner import generate_experiment_suite, ExperimentRunner
from benchmark.markup_storage import load_cache_from_yaml, export_cache_to_yaml

CACHE_FILE_PATH = "cache004.yaml"
AUDIO_PATH = "dataset/speaker004.opus"
GT_PATH = "dataset/speaker004.opus.yaml"

def custon_normalisation(text: list[str] | str) -> str:
    """Кастомная функция для нормализации."""
    if isinstance(text, list):
        return [custon_normalisation(t) for t in text]

    norm_text = text
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

def test_wer():
    """Тест jiwer, что бы посмотреть как и что выглядит"""
    reference = "мама мыла чисто раму вчера днем а папа читал газету"
    hypothesis = "мама мыла быстро раму днем папа читал вечернюю газету"

    print(f"Эталон: {reference}")
    print(f"Гипотеза: {hypothesis}")

    # Получаем детальный анализ на уровне слов
    output = jiwer.process_words(reference, hypothesis)

    print(f"Итоговый WER: {output.wer * 100:.1f}%")
    print(f"Замен (S): {output.substitutions}")
    print(f"Пропусков (D): {output.deletions}")
    print(f"Вставок (I): {output.insertions}")
    print(jiwer.visualize_alignment(output))

    # Для CER (посимвольно) используется аналогичный простой вызов:
    cer_value = jiwer.cer(reference, hypothesis)
    print(f"Итоговый CER: {cer_value * 100:.1f}%")

    # Получаем детальный объект разбора символов
    char_output = jiwer.process_characters(reference, hypothesis)

    # 1. Извлекаем конкретные числа из объекта
    substitutions = char_output.substitutions  # Замены
    insertions = char_output.insertions        # Вставки
    deletions = char_output.deletions          # Удаления
    cer = char_output.cer                      # Сам CER (0.42105...)

    print(f"CER: {cer * 100:.2f}%")
    print(f"Замены (Substitutions): {substitutions}")
    print(f"Вставки (Insertions): {insertions}")
    print(f"Удаления (Deletions): {deletions}")

    print("\n" + "="*30 + " Наглядная визуализация " + "="*30)
    # 2. Красивое выравнивание строк по символам (выведет REF, HYP и тип ошибки)
    print(jiwer.visualize_alignment(char_output))

def main():
    """Точка входа для тестирования"""
    # Проверяем наличие кэша для отладки расчетов метрик
    results = load_cache_from_yaml(CACHE_FILE_PATH)
    print(f"Попытка загрузки кэша из файла: {CACHE_FILE_PATH}")
    if results: # Файл с кэшем найден, данные загружены. При неоходимости, кэш очищается вручную
        print("Кэш загружен, бенчмарк не запускается")
        references, hypothesis = results
    else: # Кэша пока нет, запускаем бенчмарк и сохраняем результаты в кэш
        print("Кэш отсутствует, запускается бенчмарк...")
        exp_spesc = generate_experiment_suite(
            audio_path = Path(AUDIO_PATH),
            gt_file = Path(GT_PATH)
            )
        exp_spesc[0].use_oracle_vad = True
        # Отправляем exp_specs в очередь на выполнение...
        exp_runner = ExperimentRunner(exp_spesc)
        pl_results = exp_runner.run_single_combination()

        # Готовим списки эталоных фраз и гипотез
        references: list[str] = []
        hypothesis: list[str] = []
        for seg in pl_results[0].markup_segments:
            references.append(seg.text)
        for seg in pl_results[0].segments:
            hypothesis.append(seg.text)

        # Сохраняем референс и гипотезу в кэш для послежующего использования
        export_cache_to_yaml(
            file_path = CACHE_FILE_PATH,
            references = references,
            hypothesis = hypothesis
        )
        print(f"Кэш сохранен в файл: {CACHE_FILE_PATH}")

    # Создаем базовый нормализатор
    base_transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        custon_normalisation,
        jiwer.SubstituteRegexes({r"[^\w\s]": " "}), # Заменяет любой знак пунктуации на пробел
        # jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    clean_ref = base_transformation(references)
    clean_hyp = base_transformation(hypothesis)

    print(f"Количество ref: {len(references)} hyp: {len(hypothesis)}")
    output = jiwer.process_words(reference = clean_ref, hypothesis = clean_hyp)
    output_cer = jiwer.process_characters(reference = clean_ref, hypothesis = clean_hyp)

    print("========== Детализация ошибок по фразам ==========")
    # Поиск фраз с ошибками
    err_strs_count = 0
    zipped_data = zip(output.references, output.hypotheses, output.alignments)
    for idx, (_, _, alignment) in enumerate(zipped_data, start = 0):
        # В jiwer alignment — это список объектов AlignmentChunk для конкретной фразы.
        # Если в списке есть элементы с типом, отличным от 'equal', значит есть ошибки.
        has_errors = any(chunk.type != "equal" for chunk in alignment)

        if has_errors:
            err_strs_count += 1
            # Для точечной визуализации передаем конкретную пару строк
            single_output = jiwer.process_words(
                reference = [clean_ref[idx]],
                hypothesis = [clean_hyp[idx]]
            )

            # Считаем количество ошибок конкретно для этой пары
            error_count = (
                single_output.substitutions +
                single_output.deletions +
                single_output.insertions
            )

            print(f"[Фраза: {idx}] Найдено ошибок: {error_count}")

            # Печатаем красивое выравнивание
            alignment_str = jiwer.visualize_alignment(single_output, show_measures=False)
            print(alignment_str)
            print("-" * 40)

    print("\n--- ИТОГОВАЯ СТАТИСТИКА КОРПУСА ---")
    print(f"Общий WER корпуса: {output.wer:.4f}")
    print(f"Всего слов в эталоне: {output.hits + output.substitutions + output.deletions}")
    print(f"Всего замен (S): {output.substitutions}")
    print(f"Всего удалений (D): {output.deletions}")
    print(f"Всего вставок (I): {output.insertions}")
    print(f"Всего строк с ошибками: {err_strs_count} из: {len(references)}")

    print(f"Общий CER корпуса: {output_cer.cer:.4f}")
    print(f"Всего символов в эталоне: {output_cer.hits + output.substitutions + output.deletions}")
    print(f"Всего замен (S): {output_cer.substitutions}")
    print(f"Всего удалений (D): {output_cer.deletions}")
    print(f"Всего вставок (I): {output_cer.insertions}")

if __name__ == "__main__":
    main()
