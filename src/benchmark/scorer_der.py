"""
Модуль рассчитывает метрики распознавания речи:
- DER (Diarization Error Rate) - Показывает общую точность. Помогает сразу отсеять слабые модели.
"""
from collections import defaultdict
from entities import AudioSegment

def get_speaker_mapping(
    reference: list[AudioSegment],
    hypothesis: list[AudioSegment]
) -> dict[int, int]:
    """Строит маппинг ID спикеров из гипотезы в ID спикеров эталона (reference).
    
    Возвращает словарь: {hyp_speaker_id: ref_speaker_id}
    """
    # Шаг 1: Собираем матрицу пересечений по времени
    # Структура: { ref_id: { hyp_id: общая_длительность_в_секундах } }
    overlap_matrix: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for ref_seg in reference:
        if ref_seg.speaker_id is None:
            continue

        for hyp_seg in hypothesis:
            if hyp_seg.speaker_id is None:
                continue

            # Вычисляем пересечение интервалов [start, end]
            overlap_start = max(ref_seg.start_time, hyp_seg.start_time)
            overlap_end = min(ref_seg.end_time, hyp_seg.end_time)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                overlap_matrix[ref_seg.speaker_id][hyp_seg.speaker_id] += overlap_duration

    # Шаг 2: Жадный поиск максимальных совпадений
    mapping: dict[int, int] = {}
    used_ref_speakers: set[int] = set()

    # Собираем плоский список всех возможных пар и сортируем их по убыванию длительности
    all_pairs = []
    for ref_id, hyp_dict in overlap_matrix.items():
        for hyp_id, duration in hyp_dict.items():
            all_pairs.append((duration, ref_id, hyp_id))

    # Сортируем: самые длинные совпадения будут первыми
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    # Распределяем роли
    for _, ref_id, hyp_id in all_pairs:
        # Если этот спикер из гипотезы или эталона уже "занят", пропускаем парную связь
        if hyp_id in mapping or ref_id in used_ref_speakers:
            continue

        mapping[hyp_id] = ref_id
        used_ref_speakers.add(ref_id)

    return mapping

def calculate_speaker_confusion(
    reference: list[AudioSegment],
    hypothesis: list[AudioSegment],
    speaker_mapping: dict[int, int]
) -> float:
    """Высчитывает процент ошибочно определенного времени спикеров."""
    total_ref_speech_time = 0.0
    confusion_time = 0.0

    for ref_seg in reference:
        if ref_seg.speaker_id is None:
            continue

        ref_duration = ref_seg.end_time - ref_seg.start_time
        total_ref_speech_time += ref_duration

        # Ищем, что напророчила гипотеза в этот же промежуток времени
        for hyp_seg in hypothesis:
            overlap_start = max(ref_seg.start_time, hyp_seg.start_time)
            overlap_end = min(ref_seg.end_time, hyp_seg.end_time)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                # Переводим ID гипотезы в реальный ID через маппинг
                mapped_hyp_speaker = (
                    speaker_mapping.get(hyp_seg.speaker_id)
                    if hyp_seg.speaker_id is not None else None
                )

                # Если спикер не определен (Unknown) или определен неверно — это ошибка
                if mapped_hyp_speaker != ref_seg.speaker_id:
                    confusion_time += overlap_duration

    if total_ref_speech_time == 0:
        return 0.0

    return (confusion_time / total_ref_speech_time) * 100
