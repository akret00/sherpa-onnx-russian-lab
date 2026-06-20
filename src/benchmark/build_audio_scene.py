"""Модуль генерации тестовых аудиофайлов на основе сценария, исходных аудиофайлов и их разметки"""
# Запуск: PYTHONPATH=src python src/benchmark/build_audio_scene.py --input yaml_scenario_file_path
from pathlib import Path
import numpy as np
import soundfile as sf
from benchmark.entities_dataset import Scenario
from benchmark.markup_storage import load_scenario_from_yaml, load_markup_from_yaml
from entities import AudioFile
from ffmpeg_utils import read_all_samples, convert_wav_to_opus
import args_utils

# Список путей к 4 YAML-файлам разметки сегментов для 4 спикеров
MARKUP_FILES = [
    "dataset/speaker001.opus.yaml",
    "dataset/speaker002.opus.yaml",
    "dataset/speaker003.opus.yaml",
    "dataset/speaker004.opus.yaml"
]

# Имя выходного файла сценария (можно генерировать динамически или захардкодить)
OUTPUT_SCENE_BASE_PATH = "dataset"

def build_audio_scene(
    scenario: Scenario,
    audio_files: list[AudioFile],
    src_audio_data: list[np.ndarray]
) -> np.ndarray:
    """Собирает единый аудиомассив сценария из исходных аудиоданных спикеров.

    Args:
        scenario: Объект сценария, содержащий эпизоды и события.
        audio_files: Список метаданных файлов. Индекс равен (file_id - 1).
        src_audio_data: Список массивов numpy (PCM float32) исходных записей.

    Returns:
        np.ndarray: Финальный аудиомассив в формате float32, готовый к записи в WAV.
    """
    # 1. Проверяем, есть ли вообще эпизоды в сценарии
    if not scenario.episodes:
        print("Сценарий пуст, возвращаем пустой аудиомассив.")
        return np.zeros(0, dtype=np.float32)

    sample_rate = scenario.sample_rate

    # 2. Определяем общую длину итогового аудиофайла.
    # Находим самое позднее время окончания (end) среди абсолютно всех событий.
    max_end_time = 0.0
    for episode in scenario.episodes:
        for event in episode.events:
            if event.end > max_end_time:
                max_end_time = event.end

    # Переводим секунды в количество сэмплов (индексы массива)
    total_samples = int(np.ceil(max_end_time * sample_rate))

    # Создаем пустой результирующий массив, заполненный тишиной (нолями)
    result_audio = np.zeros(total_samples, dtype=np.float32)

    # 3. Последовательно перебираем эпизоды и события сценария
    for episode in scenario.episodes:

        # --- ТУТ ДОЛЖНО БЫТЬ НАЛОЖЕНИЕ ШУМОВ НА УРОВНЕ ЭПИЗОДА ---
        # Если в будущем понадобится добавить фоновый шум (например, episode.noise),
        # логику его подмешивания (с учетом длины эпизода) нужно будет внедрить здесь.
        # --------------------------------------------------------

        for event in episode.events:
            # Индекс в списках на единицу меньше, чем ID (так как ID начинаются с 1)
            file_idx = event.file_id - 1

            # Извлекаем метаданные файла и сами аудиоданные нужного спикера
            meta_file = audio_files[file_idx]
            raw_audio = src_audio_data[file_idx]

            # Ищем исходный сегмент по его segment_id внутри метаданных файла
            src_segment = None
            if meta_file.segments:
                for seg in meta_file.segments:
                    if seg.id == event.segment_id:
                        src_segment = seg
                        break

            if src_segment is None:
                raise ValueError(
                    f"Сегмент с ID {event.segment_id} не найден в файле {event.file_id}"
                )

            # Переводим временные границы ИСХОДНОГО сегмента в сэмплы
            src_start_sample = int(round(src_segment.start_time * sample_rate))
            src_end_sample = int(round(src_segment.end_time * sample_rate))

            # Вырезаем чистую фразу из исходного аудио спикера
            phrase_chunk = raw_audio[src_start_sample:src_end_sample]

            # Рассчитываем целевую (требуемую сценарием) длительность в сэмплах
            target_duration_samples = int(round((event.end - event.start) * sample_rate))

            # Подгоняем длину куска под требования сценария (обрезаем или дополняем тишиной)
            current_len = len(phrase_chunk)
            if current_len > target_duration_samples:
                # Обрезаем сегмент справа, если он длиннее, чем нужно сценарию
                phrase_chunk = phrase_chunk[:target_duration_samples]
            elif current_len < target_duration_samples:
                # Дополняем тишиной справа, если он короче
                padding = np.zeros(target_duration_samples - current_len, dtype=np.float32)
                phrase_chunk = np.concatenate([phrase_chunk, padding])

            # Применяем изменение громкости (Gain), если оно задано (не равно 0.0)
            if event.gain_db != 0.0:
                # Формула перевода децибел в линейный коэффициент амплитуды
                gain_factor = 10 ** (event.gain_db / 20.0)
                phrase_chunk = phrase_chunk * gain_factor

            # Определяем, куда именно в результирующий файл вставить этот кусок
            target_start_sample = int(round(event.start * sample_rate))
            target_end_sample = target_start_sample + target_duration_samples

            # Важная строка: используем "+=" вместо "=".
            # Благодаря этому, если временные отрезки разных событий пересекаются (overlap),
            # голоса спикеров автоматически и правильно смешаются (наложатся друг на друга).
            result_audio[target_start_sample:target_end_sample] += phrase_chunk

    # 4. Защита от клиппинга (Пиковая нормализация)
    # Если при наложении голосов амплитуда превысила допустимый максимум (1.0 или -1.0)
    max_peak = np.max(np.abs(result_audio))
    if max_peak > 1.0:
        # Пропорционально уменьшаем громкость всего трека, оставляя запас в 1% до лимита
        result_audio = result_audio * (0.99 / max_peak)
        print(f"Внимание: Обнаружен клиппинг (пик {max_peak:.2f}). Трек нормализован.")

    return result_audio

def save_audiodata(
    file_path: Path,
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> None:
    """Сохраняет аудиомассив numpy в файл формата WAV с помощью библиотеки soundfile.

    Автоматически конвертирует float32 в стандартный 16-битный PCM для 
    максимальной совместимости с пайплайнами распознавания речи.

    Args:
        file_path: Путь к итоговому WAV файлу (объект Path).
        audio_data: Аудиоданные в формате PCM (numpy.ndarray с типом float32).
        sample_rate: Частота дискретизации аудио (по умолчанию 16000 Гц).
    """
    # Гарантируем, что папка для файла существует
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Записываем файл на диск.
    # subtype='PCM_16' принудительно сохраняет файл как классический 16-bit WAV.
    # Библиотека soundfile сама правильно масштабирует float32 [-1.0, 1.0] в int16.
    sf.write(
        file=str(file_path),
        data=audio_data,
        samplerate=sample_rate,
        subtype='PCM_16'
    )

    duration = len(audio_data) / sample_rate
    print(f"Аудио успешно сохранено: {file_path} (Длительность: {duration:.2f} сек.)")


def main():
    """Точка входа скрипта генерации аудиосцены."""
    print("Запуск генератора сцены на основе сценария...")

    # 1. Получение пути к файлу с рецептом из аргументов командной строки
    args = args_utils.parse_args()

    if not args.input:
        raise ValueError("Не указан путь к YAML файлу со сценарием")
    yaml_scenario_path = Path(args.input)

    print(f"Загрузка сценария из: {yaml_scenario_path}")

    # 2. Загрузка сценария в датакласс Scenarion
    scenario = load_scenario_from_yaml(yaml_scenario_path)
    print(
        f"Сценарий '{yaml_scenario_path}' успешно загружен. "
        f"Эпизодов в сценарии: {len(scenario.episodes)}"
    )

    # 3. Загрузка разметки сегментов для всех четырех аудиофайлов
    audio_files: list[AudioFile] = []
    print(f"Загрузка {len(MARKUP_FILES)} файлов разметки спикеров...")

    for path_str in MARKUP_FILES:
        yaml_path = Path(path_str)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Файл разметки не найден: {yaml_path.resolve()}")

        # load_markup_from_yaml возвращает tuple[list[Speaker], AudioFile]
        # Нам нужен только второй элемент (AudioFile)
        _, audio_file = load_markup_from_yaml(yaml_path)
        audio_files.append(audio_file)

        # Небольшая валидация, что внутри AudioFile есть сегменты разметки
        segments_count = len(audio_file.segments) if audio_file.segments else 0
        print(f" -> Загружен файл для спикера. Сегментов найдено: {segments_count}")

    # 4. Загрузка исходных аудиоданных
    print(f"Загрузка {len(audio_files)} исходных файлов аудиоданых...")
    src_audio_data = []
    for af in audio_files:
        audio_file_path = af.file_path
        src_audio_data.append(read_all_samples(audio_file_path))
        print(f" -> Загружен аудиофайл {audio_file_path}")

    # 5. Генерация аудио сцены
    print("Генерация аудио сцены ...")
    audio_scene = build_audio_scene(
        scenario = scenario,
        audio_files = audio_files,
        src_audio_data = src_audio_data
    )

    # Модифицируем имя выходного файла на основе ID рецепта, чтобы не затирать старые
    output_path = Path(f"{OUTPUT_SCENE_BASE_PATH}/{yaml_scenario_path.stem}.wav")

    # 6. Запись сгенерированной аудиосцены в wav файл
    print(f"Запись сгенерированной аудиосцены в wav файл: {output_path}")
    save_audiodata(output_path, audio_scene)
    print(f"Аудиосцена успешно сохранена в wav файл: {output_path}")

    # 7. Конвертация wav файла в opus
    print(f"Конвертация wav файла {output_path} в opus формат")
    opus_path = convert_wav_to_opus(wav_path = output_path)
    print(f"Wav файл успешно сконвертировано в opus файл: {opus_path}")


if __name__ == "__main__":
    main()
