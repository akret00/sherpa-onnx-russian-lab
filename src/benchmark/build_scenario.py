"""Модуль для генерации сценариев на основе рецепта"""
# Запуск: PYTHONPATH=src python src/benchmark/build_scenario.py --input yaml_recipe_file_path
from pathlib import Path
from benchmark.dataset_entities import (
    AudioSegmentMarkup,
    AudioFileMarkup,
    Recipe,
    Scenario, ScenarioEpisode, ScenarioEvent,
    NoiseConfig,
)
from benchmark.dataset_storage import (
    load_recipe_from_yaml,
    load_markup_from_yaml,
    export_scenario_to_yaml,
)
import args_utils

# Константа для фразы-заглушки (можно вынести в настройки)
BACKUP_PHRASE_ID = 59 # Секунду… я отвлёкся. Продолжаем.

# Список путей к 4 YAML-файлам разметки сегментов для 4 спикеров
MARKUP_FILES = [
    "dataset/speaker001.opus.yaml",
    "dataset/speaker002.opus.yaml",
    "dataset/speaker003.opus.yaml",
    "dataset/speaker004.opus.yaml"
]

# Имя выходного файла сценария (можно генерировать динамически или захардкодить)
OUTPUT_SCENARIO_BASE_PATH = "dataset"

def generate_scenario(recipe: Recipe, audio_files: list[AudioFileMarkup]) -> Scenario:
    """
    Генерирует сценарий на основе рецепта и списка аудио файлов
    В audio_files должен быть список AudioFileMarkup в порядке: индекс в массиве == speaker_id - 1
    То есть AudioFileMarkup должны идти в порядке для speaker1, speaker2, speaker3, speaker4
    """
    # 1. Индексация данных для быстрого поиска: (speaker_id, phrase_id) -> AudioSegmentMarkup
    # Предполагаем, что в каждом AudioFileMarkup.segments лежат объекты AudioSegmentMarkup
    phrase_index: dict[int, dict[int, AudioSegmentMarkup]] = {}

    for audio_file in audio_files:
        # Так как speaker_id находитс внутри сегментов, берем его оттуда, считая, что все
        # сегменты одного аудиофайла относятся к одному и тому же спикеру
        if not audio_file.segments:
            continue

        first_segment = audio_file.segments[0]
        # Если speaker_id не проставлен, используем логический ID на основе
        # позиции в списке (индекс + 1)
        if first_segment.speaker_id is not None:
            spk_id = first_segment.speaker_id
        else:
            spk_id = audio_files.index(audio_file) + 1

        phrase_index[spk_id] = {}
        for segment in audio_file.segments:
            if isinstance(segment, AudioSegmentMarkup) and segment.phrase_id is not None:
                phrase_index[spk_id][segment.phrase_id] = segment

    # 2. Определение доступных спикеров (их ровно по количеству файлов, например, 4)
    available_speakers = sorted(list(phrase_index.keys()))
    num_speakers = len(available_speakers) # Ожидаем 4

    # Создаем пустоя список эпизодов сценария
    episodes: list[ScenarioEpisode] = []
    pause_seconds = recipe.pause_ms / 1000.0

    # 3. Генерация 4 вариантов ротации
    current_time = 0.00
    for rotation_idx in range(num_speakers):
        # Строим маппинг: dictor_id -> speaker_id
        # Пример для rotation_idx=0 (Вариант 1): D1->Spk1, D2->Spk2, D3->Spk3, D4->Spk4
        # Пример для rotation_idx=1 (Вариант 2): D1->Spk2, D2->Spk3, D3->Spk4, D4->Spk1
        dictor_to_speaker: dict[int, int] = {}
        for d_idx in range(num_speakers):
            dictor_id = d_idx + 1 # Дикторы 1..4
            # Вычисляем сдвиг циклической ротации и ИД спикера для него
            speaker_idx = (d_idx + rotation_idx) % num_speakers
            dictor_to_speaker[dictor_id] = available_speakers[speaker_idx]

        # Собираем события для текущей сцены
        scene_events: list[ScenarioEvent] = []
        # current_time = 0.00 # Тут это будет насинать таймлайн для каждого эпизода с нуля

        # 4. Трансляция таймлайна рецепта
        for step in recipe.timeline:
            target_speaker_id = dictor_to_speaker[step.dictor_id]
            speaker_phrases = phrase_index.get(target_speaker_id, {})

            # Поиск фразы или подстановка заглушки
            segment = speaker_phrases.get(step.phrase_id)
            if not segment:
                segment = speaker_phrases.get(BACKUP_PHRASE_ID)
                if not segment:
                    print(
                        f"[Warning] Speaker {target_speaker_id} misses both phrase "
                        "{step.phrase_id} and backup {BACKUP_PHRASE_ID}. Skipping step."
                    )
                    continue

            # Расчет длительности исходного сегмента
            duration = segment.end_time - segment.start_time

            # Создаем событие сценария
            # В поле file_id пишем target_speaker_id (так как 1 файл = 1 спикер)
            event = ScenarioEvent(
                file_id = target_speaker_id,
                segment_id = segment.id if segment.id is not None else 0,
                start = round(current_time, 2), # Нужно ли тут округление до 10 миллисекунд?
                end = round(current_time + duration, 2),
                text = segment.text,
                gain_db=0.0
            )
            scene_events.append(event)

            # Расчет времени для следующего шага нарастающим итогом
            current_time += duration + pause_seconds

        # Формируем итоговую сцену для этого варианта ротации
        scene_id = f"{recipe.recipe_id}_rot{rotation_idx + 1}"
        scene_desc = (
            f"Rotation {rotation_idx + 1}: "
            + ", ".join(
                [
                f"D{d_id}->Spk{s_id}"
                for d_id, s_id in dictor_to_speaker.items()
                ]
            )
        )

        episode = ScenarioEpisode(
            id=scene_id,
            description=scene_desc,
            events=scene_events,
            noise=NoiseConfig(enabled=False)
        )
        episodes.append(episode)

    # 5. Сборка финального датасета сценариев
    return Scenario(episodes = episodes)

def main() -> None:
    """Точка входа скрипта генерации сценария."""
    print("Запуск генератора сценариев...")

    # 1. Получение пути к файлу с рецептом из аргументов командной строки
    args = args_utils.parse_args()

    if not args.input:
        raise ValueError("Не указан путь к YAML файлу с рецептом")
    yaml_recipe_path = Path(args.input)

    print(f"Загрузка рецепта из: {yaml_recipe_path}")

    # 2. Загрузка рецепта в датакласс Recipe
    recipe = load_recipe_from_yaml(str(yaml_recipe_path))
    print(
        f"Рецепт '{recipe.recipe_id}' успешно загружен. "
        f"Шагов в таймлайне: {len(recipe.timeline)}"
    )

    # 3. Загрузка разметки сегментов для всех четырех аудиофайлов
    audio_files: list[AudioFileMarkup] = []
    print(f"Загрузка {len(MARKUP_FILES)} файлов разметки спикеров...")

    for path_str in MARKUP_FILES:
        yaml_path = Path(path_str)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Файл разметки не найден: {yaml_path.resolve()}")

        # load_markup_from_yaml возвращает tuple[list[Speaker], AudioFileMarkup]
        # Нам нужен только второй элемент (AudioFileMarkup)
        _, audio_file = load_markup_from_yaml(yaml_path)
        audio_files.append(audio_file)

        # Небольшая валидация, что внутри AudioFileMarkup есть сегменты разметки
        segments_count = len(audio_file.segments) if audio_file.segments else 0
        print(f" -> Загружен файл для спикера. Сегментов найдено: {segments_count}")

    # 4. Генерация сценария (с учетом циклической ротации и таймингов)
    print("Генерация эпизодов сценария и расчет таймлайна...")
    scenario = generate_scenario(recipe, audio_files)

    # Модифицируем имя выходного файла на основе ID рецепта, чтобы не затирать старые
    output_path = f"{OUTPUT_SCENARIO_BASE_PATH}/scenario_{yaml_recipe_path.stem}.yaml"

    # 5. Экспорт полученного объекта типа Scenario в YAML
    print(f"Экспорт готового сценария в файл: {output_path}")
    export_scenario_to_yaml(output_path, scenario)

    print(f"Успешно! Сгенерировано эпизодов: {len(scenario.episodes or [])}")


if __name__ == "__main__":
    main()
