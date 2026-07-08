"""
Модуль рассчитывает метрики распознавания речи:
- DER (Diarization Error Rate) - Показывает общую точность. Помогает сразу отсеять слабые модели.
"""
from collections import defaultdict
from entities import AudioSegment
from benchmark.experiment_entities import PipelineResultExperiment, MetricExpDER

def get_speaker_mapping(
    reference: list[AudioSegment],
    hypothesis: list[AudioSegment]
) -> dict[int, int]:
    """
    Строит маппинг ID спикеров из гипотезы в ID спикеров эталона (reference).
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

def calculate_speaker_confusion(pl_res_exp: PipelineResultExperiment) -> MetricExpDER:
    """
    Высчитывает процент ошибочно определенного времени спикеров.
    Расчет производится посегментно и игнорирует ошибки VAD
    """
    # Получаем маппинг спикеров з эталона и гипотезы
    speaker_mapping: dict[int, int] = get_speaker_mapping(
        reference = pl_res_exp.markup_segments, hypothesis = pl_res_exp.segments
    )
    total_ref_speech_time = 0.0
    confusion_time = 0.0

    for ref_seg in pl_res_exp.markup_segments:
        if ref_seg.speaker_id is None:
            continue

        ref_duration = ref_seg.end_time - ref_seg.start_time
        total_ref_speech_time += ref_duration

        # Ищем, что напророчила гипотеза в этот же промежуток времени
        for hyp_seg in pl_res_exp.segments:
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

    scr = (confusion_time / total_ref_speech_time) if total_ref_speech_time != 0 else None
    if pl_res_exp.exp_spec.use_oracle_vad:
        scr_oracle_vad = scr
        scr_evaluated_vad = None
    else:
        scr_oracle_vad = None
        scr_evaluated_vad = scr

    total_der = MetricExpDER(
        obj_id = pl_res_exp.exp_id,
        confusion_time = confusion_time,
        total_ref_speech_time = total_ref_speech_time,
        scr_evaluated_vad = scr_evaluated_vad,
        scr_oracle_vad = scr_oracle_vad,
    )

    return total_der


def calculate_frame_based_der(
    pl_res_exp: PipelineResultExperiment,
    frame_step: float = 0.1
) -> MetricExpDER:
    """Расчитывает покадровый DER (включая FA, MS и Confusion)."""
    # 1. Получаем маппинг спикеров из эталона и гипотезы
    speaker_mapping: dict[int, int] = get_speaker_mapping(
        reference=pl_res_exp.markup_segments,
        hypothesis=pl_res_exp.segments
    )

    # 2. Определяем общую длину аудио для создания временной сетки
    # Находим максимальную временную точку среди эталона и гипотезы
    max_time = 0.0
    for seg in pl_res_exp.markup_segments + pl_res_exp.segments:
        if seg.end_time > max_time:
            max_time = seg.end_time

    total_frames = int(max_time / frame_step) + 1

    # Инициализируем массивы для кадров (None означает тишину)
    ref_frames = [None] * total_frames
    hyp_frames = [None] * total_frames

    # 3. Заполняем сетку эталона (Reference)
    for ref_seg in pl_res_exp.markup_segments:
        # Если в самом эталоне спикер None, трактуем это как тишину/неразмеченный кусок
        if ref_seg.speaker_id is None:
            continue
        start_idx = int(ref_seg.start_time / frame_step)
        end_idx = int(ref_seg.end_time / frame_step)
        for i in range(start_idx, end_idx):
            if 0 <= i < total_frames:
                ref_frames[i] = ref_seg.speaker_id

    # 4. Заполняем сетку гипотезы (Hypothesis) с учетом уникальных технических ID для None
    # Каждому физическому сегменту с None присваиваем уникальный маркер "UNKNOWN_{idx}"
    for idx, hyp_seg in enumerate(pl_res_exp.segments):
        start_idx = int(hyp_seg.start_time / frame_step)
        end_idx = int(hyp_seg.end_time / frame_step)

        # Если спикер None, генерируем для этого конкретного сегмента уникальную строку
        if hyp_seg.speaker_id is None:
            hyp_speaker = f"UNKNOWN_{idx}"
        else:
            # Сразу применяем маппинг, переводя гипотетический ID в эталонный
            hyp_speaker = speaker_mapping.get(hyp_seg.speaker_id)

        for i in range(start_idx, end_idx):
            if 0 <= i < total_frames:
                hyp_frames[i] = hyp_speaker

    # 5. Считаем метрики по кадрам
    total_ref_speech_frames = 0
    missed_speech_frames = 0
    false_alarm_frames = 0
    confusion_frames = 0

    for r_spk, h_spk in zip(ref_frames, hyp_frames):
        is_ref_speech = r_spk is not None
        is_hyp_speech = h_spk is not None

        if is_ref_speech:
            total_ref_speech_frames += 1

        # Сценарий А: В эталоне говорят, в гипотезе — тишина (пропуск речи)
        if is_ref_speech and not is_hyp_speech:
            missed_speech_frames += 1

        # Сценарий Б: В эталоне тишина, в гипотезе система нашла речь (ложная тревога)
        elif not is_ref_speech and is_hyp_speech:
            false_alarm_frames += 1

        # Сценарий В: Речь есть и там, и там, но спикеры не совпали (путаница)
        # Сюда же автоматически прилетят строки "UNKNOWN_XYZ", так как они никогда не равны r_spk
        elif is_ref_speech and is_hyp_speech:
            if r_spk != h_spk:
                confusion_frames += 1

    # Переводим кадры обратно в секунды
    total_ref_speech_time = total_ref_speech_frames * frame_step
    missed_speech_time = missed_speech_frames * frame_step
    false_alarm_time = false_alarm_frames * frame_step
    confusion_time = confusion_frames * frame_step

    total_error_time = missed_speech_time + false_alarm_time + confusion_time

    scr = (confusion_time / total_ref_speech_time) if total_ref_speech_time != 0 else None
    der = (total_error_time / total_ref_speech_time) if total_ref_speech_time != 0 else None
    if pl_res_exp.exp_spec.use_oracle_vad:
        scr_oracle_vad = scr
        scr_evaluated_vad = None
        der_oracle_vad = der
        der_evaluated_vad = None
    else:
        scr_oracle_vad = None
        scr_evaluated_vad = scr
        der_oracle_vad = None
        der_evaluated_vad = der

    return MetricExpDER(
        obj_id = pl_res_exp.exp_id,
        confusion_time = confusion_time,
        total_error_time = total_error_time,
        total_ref_speech_time = total_ref_speech_time,
        # SCR в рамках этой функции — это посекундная путаница к общему времени речи
        scr_evaluated_vad = scr_evaluated_vad,
        scr_oracle_vad = scr_oracle_vad,
        # Классический DER
        der_evaluated_vad = der_evaluated_vad,
        der_oracle_vad = der_oracle_vad
    )
