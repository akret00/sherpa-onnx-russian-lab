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
# Запуск: PYTHONPATH=src python src/benchmark/calc_wer_metrics.py --input yaml_recipe_file_path
from pathlib import Path
import jiwer
from benchmark.benchmark_runner import generate_benchmark_suite, BenchmarkRunner

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
    bm_spesc = generate_benchmark_suite(
        audio_path = Path("dataset/speaker002.opus"),
        gt_file = Path("dataset/speaker002.opus.yaml")
        )
    bm_spesc[0].use_oracle_vad = True
    # Отправляем specs_pool в пул потоков/процессов на выполнение...
    bm_runner = BenchmarkRunner(bm_spesc)
    pl_results = bm_runner.run_single_combination()

    # Готовим списки эталоных фраз и гипотез
    references: list[str] = []
    hypothesis: list[str] = []
    for seg in pl_results[0].segments:
        references.append(seg.text)
    for seg in pl_results[0].markup_segments:
        hypothesis.append(seg.text)

    print(f"Количество ref: {len(references)} hyp: {len(hypothesis)}")
    output = jiwer.process_words(reference = references, hypothesis = hypothesis)
    output_cer = jiwer.process_characters(reference = references, hypothesis = hypothesis)

    print("========== Детализация ошибок по фразам ==========")
    # Поиск фраз с ошибками
    zipped_data = zip(output.references, output.hypotheses, output.alignments)
    for idx, (ref, hyp, alignment) in enumerate(zipped_data, start = 0):
        # В jiwer alignment — это список объектов AlignmentChunk для конкретной фразы.
        # Если в списке есть элементы с типом, отличным от 'equal', значит есть ошибки.
        has_errors = any(chunk.type != "equal" for chunk in alignment)

        if has_errors:
            # Для точечной визуализации передаем конкретную пару строк
            single_output = jiwer.process_words(
                reference = [references[idx]],
                hypothesis = [hypothesis[idx]]
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

    print(f"Общий CER корпуса: {output_cer.cer:.4f}")
    print(f"Всего символов в эталоне: {output_cer.hits + output.substitutions + output.deletions}")
    print(f"Всего замен (S): {output_cer.substitutions}")
    print(f"Всего удалений (D): {output_cer.deletions}")
    print(f"Всего вставок (I): {output_cer.insertions}")

if __name__ == "__main__":
    main()
