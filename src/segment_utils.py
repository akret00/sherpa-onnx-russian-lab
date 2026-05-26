"""Модуль содержит утилиты для работы с сегментами речи"""
import numpy
import config

# Словарь профилей (инструмент настроек)
PROFILES = {
    "quiet_room": {"threshold": 0.01, "smooth_window_ms": 10},
    "low_noise":  {"threshold": 0.04, "smooth_window_ms": 30}, # Профиль по умолчанию
    "street_noise":{"threshold": 0.08, "smooth_window_ms": 60}
}

def visualize_segment_energy(
    audio_segment: numpy.ndarray,
    num_bars: int = 40,
    ):
    """
    Визуализирует распределение энергии внутри короткого аудиосегмента в консоли.
    audio_segment: одномерный массив numpy (float32 или int16)
    """
    if len(audio_segment) == 0:
        print("[Пустой сегмент]")
        return

    # Приводим к float32 для корректного расчета энергии, если аудио в int16
    if audio_segment.dtype != numpy.float32:
        audio = audio_segment.astype(numpy.float32) / 32768.0
    else:
        audio = audio_segment

    # Бьем сегмент на равные чанки для построения колонок графика
    chunks = numpy.array_split(audio, num_bars)

    # Считаем RMS (среднеквадратичную амплитуду) для каждого чанка
    energies = []
    for chunk in chunks:
        if len(chunk) > 0:
            rms = numpy.sqrt(numpy.mean(chunk**2))
            energies.append(rms)
        else:
            energies.append(0.0)

    energies = numpy.array(energies)
    max_energy = numpy.max(energies)

    if max_energy == 0:
        print("[Абсолютная тишина]")
        return

    # Нормализуем энергию для красивого отображения в консоли (высота столбика до 10 символов)
    max_bar_height = 10
    normalized_energies = (energies / max_energy * max_bar_height).astype(int)

    # Символы для построения вертикального графика (блоки разной высоты)
    # Если в консоли не поддерживается UTF-8, можно заменить на '#'
    bar_chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█", "█"]

    print("\n" + "="*50)
    duration = len(audio_segment) / config.SR
    print(f"АНАЛИЗ СЕГМЕНТА | Длина: {duration:.3f} сек. | Сэмплов: {len(audio_segment)}")
    print("="*50)

    # Рисуем график построчно сверху вниз
    for level in range(max_bar_height, 0, -1):
        line = ""
        for h in normalized_energies:
            if h >= level:
                line += "█"
            elif h == level - 1 and level > 1:
                # Рисуем промежуточный символ для плавности, если консоль поддерживает
                line += "▄"
            else:
                line += " "
        print(line)

    print("-" * num_bars + "  <- Временная шкала (от начала к концу)")

def trim_silence_dyn_end(
    audio_segment: numpy.ndarray,
    profile_name="low_noise",
    pad_ms: int = 100
) -> numpy.ndarray:
    """
    Динамически находит конец речи и обрезает тишину, оставляя безопасный зазор.
    """
    if len(audio_segment) == 0:
        return audio_segment

    if audio_segment.dtype != numpy.float32:
        audio = audio_segment.astype(numpy.float32) / 32768.0
    else:
        audio = audio_segment

    cfg = PROFILES.get(profile_name, PROFILES["low_noise"])
    threshold = cfg["threshold"]
    smooth_window_ms = cfg["smooth_window_ms"]

    # 1. Проверка длины: если фраза длинная, можно ослабить правила
    duration = len(audio) / config.SR
    if duration > 2.0:
        # Например, снижаем порог для длинных фраз, чтобы перестраховаться
        current_threshold = threshold * 0.5
    else:
        current_threshold = threshold

    # 2. Сглаживание скользящим окном (чтобы не резать по одиночным отсчетам)
    window_size = int(config.SR * (smooth_window_ms / 1000.0))
    # Простейшее сглаживание амплитуды методом свёртки
    smoothed_amplitude = numpy.convolve(
        numpy.abs(audio), numpy.ones(window_size)/window_size, mode='same'
    )

    # 3. Поиск конца речи
    indices = numpy.where(smoothed_amplitude > current_threshold)[0]
    if len(indices) == 0:
        return audio # Возвращаем как есть, если звук слишком тихий

    last_speech_idx = indices[-1]

    # 4. Добавление защитного хвоста в 100 мс
    tail_samples = int(config.SR * (pad_ms / 1000.0))
    cut_idx = min(last_speech_idx + tail_samples, len(audio))

    # print(f"Исходная длина: {len(audio)} | Новая длина: {cut_idx} (сэмплов)")
    return audio[:cut_idx]

def trim_silence_abs_end(
    audio_segment: numpy.ndarray,
    threshold: float = 0.03,
    pad_ms: int = 100
) -> numpy.ndarray:
    """
    Динамически находит конец речи и обрезает тишину, оставляя безопасный зазор.
    """
    if len(audio_segment) == 0:
        return audio_segment

    if audio_segment.dtype != numpy.float32:
        audio = audio_segment.astype(numpy.float32) / 32768.0
    else:
        audio = audio_segment

    # 1. Находим абсолютные значения амплитуды
    abs_signal = numpy.abs(audio)

    # 2. Ищем индекс последнего элемента, который ПРЕВЫШАЕТ порог
    # numpy.where возвращает индексы; берем [-1] для получения самого последнего
    indices = numpy.where(abs_signal > threshold)[0]

    # Если весь сегмент тише порога, возвращаем пустой массив или оригинал
    if len(indices) == 0:
        return numpy.array([], dtype=audio.dtype)

    last_speech_idx = indices[-1]

    # 3. Расчитываем, сколько семплов нужно оставить для хвоста (100 мс)
    tail_samples = int(config.SR * (pad_ms / 1000.0))

    # 4. Вычисляем финальный индекс среза
    cut_idx = last_speech_idx + 1 + tail_samples

    # Ограничиваем индекс, чтобы не выйти за пределы исходного массива
    cut_idx = min(cut_idx, len(audio))

    # 5. Возвращаем обрезанный массив
    # print(f"Исходная длина: {len(audio)} | Новая длина: {cut_idx} (сэмплов)")
    return audio[:cut_idx]

def trim_silence_fix_end(
    audio_segment: numpy.ndarray,
) -> numpy.ndarray:
    """
    Обрезает 0.5 секунд тишины спава.
    """
    # 0.5 секунды умножаем на частоту дискретизации
    samples_to_cut = int(config.SR * 0.5)

    # Защита: если весь файл короче 0.5 секунды, возвращаем пустой массив
    if len(audio_segment) <= samples_to_cut:
        return numpy.array([], dtype=audio_segment.dtype)

    # Отрезаем кусок справа с помощью стандартного среза NumPy
    return audio_segment[:-samples_to_cut]
