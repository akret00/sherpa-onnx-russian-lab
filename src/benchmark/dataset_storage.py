"""Модуль содержит функции выгрузки и загрузки результатов разметки в YAML формат"""
from pathlib import Path
import os
import dataclasses
import yaml
from entities import Speaker
from benchmark.dataset_entities import (
    AudioSegmentMarkup, AudioFileMarkup, Recipe, RecipeStep,
    Scenario, ScenarioEpisode, ScenarioEvent, NoiseConfig
)

def export_markup_to_yaml(
    yaml_path: str | Path,
    speakers: list[Speaker] | None,
    audio_file: AudioFileMarkup,
) -> None:
    """Экспорт датаклассов в чистый человекочитаемый YAML."""
    default_spk_id = 1
    default_file_id = 1
    default_notes = "laptop mic"

    # 1. Формируем секцию speakers (убираем эмбеддинги и счетчики)
    yaml_speakers = []
    if speakers is not None: # Если есть список спикеров из YAML файла
        for spk in speakers:
            yaml_speakers.append({"id": spk.id, "name": spk.name, "notes": default_notes})
    else: # Если список спикеров из YAML не грузился, то формируем одного спикера
        yaml_speakers.append(
            {"id": default_spk_id, "name": default_file_id, "notes": default_notes}
        )

    # 2. Формируем секцию files
    yaml_files = []
    # yaml_files.append({"id": file.id, "file_path": file.file_path})
    yaml_files.append(
        {
            "id": audio_file.id if audio_file.id is not None else default_file_id,
            "file_path": audio_file.file_path
        }
    )

    # 3. Формируем секцию segments, собирая их из всех файлов
    yaml_segments = []
    if audio_file.segments:
        seg_id = 0
        for seg in audio_file.segments:
            seg_id += 1
            yaml_segments.append(
                {
                    "id": seg_id, # Перенумерация id сегментов с начала
                    "speaker_id": seg.speaker.id if seg.speaker.id is not None else default_spk_id,
                    "file_id": audio_file.id if audio_file.id is not None else default_file_id,
                    "phrase_id": seg.phrase_id,
                    "text": seg.text,
                    "speech_start": round(seg.start_time, 2),
                    "speech_end": round(seg.end_time, 2),
                }
            )

    # Собираем финальный документ
    data_to_save = {
        "dataset_version": audio_file.dataset_version,
        "sample_rate": "16000",
        "speakers": yaml_speakers,
        "files": yaml_files,
        "segments": yaml_segments,
    }

    # Сохраняем с отключением дефолтных python-тегов и красивыми отступами
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data_to_save,
            f,
            allow_unicode = True,
            sort_keys = False,
            default_flow_style = False,
        )


def load_markup_from_yaml(yaml_path: str | Path) -> tuple[list[Speaker], AudioFileMarkup]:
    """Загрузка из YAML и восстановление связей в датаклассах."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 1. Восстанавливаем спикеров
    speakers_dict = {}
    for spk_data in data.get("speakers", []):
        spk = Speaker(
            id = int(spk_data["id"]),
            name = spk_data["name"]
        )
        speakers_dict[spk.id] = spk

    # 2. Восстанавливаем файлы
    file_data = data.get("files", [])[0]
    audio_file = AudioFileMarkup(
        id = int(file_data["id"]),
        file_path = file_data["file_path"],
        dataset_version = data["dataset_version"],
        segments = []
    )

    # 3. Восстанавливаем сегменты и связываем их
    for seg_data in data.get("segments", []):
        f_id = int(seg_data["file_id"])
        spk_id = int(seg_data["speaker_id"])

        speaker = speakers_dict.get(spk_id)

        segment = AudioSegmentMarkup(
            id = int(seg_data["id"]),
            audio_file_id = f_id,
            audio_file = audio_file,  # циклическая ссылка обратно на файл
            speaker_id = spk_id,
            speaker = speaker,  # ссылка на объект спикера
            start_time = float(seg_data["speech_start"]),
            end_time = float(seg_data["speech_end"]),
            phrase_id = int(seg_data.get("phrase_id", seg_data["id"])),
            text = seg_data["text"],
            word_count = len(seg_data["text"].split()) if seg_data["text"] else 0,
        )

        # Добавляем сегмент в список родительского файла
        if audio_file and audio_file.segments is not None:
            audio_file.segments.append(segment)

    return list(speakers_dict.values()), audio_file

def load_recipe_from_yaml(file_path: str) -> Recipe:
    """Загружает рецепт из YAML-файла и возвращает объект Recipe.
    Args:
        file_path: Относительный или абсолютный путь к YAML-файлу.
    Returns:
        Объект Recipe с валидированными данными и списком шагов внутри.
    """
    # Нормализуем путь относительно текущей рабочей директории
    full_path = os.path.abspath(file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Файл рецепта не найден по пути: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        # Loader=yaml.SafeLoader защищает от выполнения произвольного кода из YAML
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if not data:
        raise ValueError(f"Файл рецепта {full_path} пуст")

    # Извлекаем шаги таймлайна и преобразуем их в объекты RecipeStep
    raw_timeline = data.get("timeline", [])
    timeline_steps = [
        RecipeStep(
            step_id=int(step["step_id"]),
            dictor_id=int(step["dictor_id"]),
            phrase_id=int(step["phrase_id"]),
        )
        for step in raw_timeline
    ]

    # Собираем и возвращаем финальный объект Recipe
    return Recipe(
        recipe_id=str(data["recipe_id"]),
        description=str(data.get("description", "")),
        pause_ms=int(data.get("pause_ms", 0)),
        timeline=timeline_steps,
    )

def load_scenario_from_yaml(file_path: str) -> Scenario:
    """Загружает сценарий из YAML-файла и возвращает заполненный объект Scenario.
    Args:
        file_path: Относительный или абсолютный путь к YAML-файлу сценария.
    Returns:
        Объект Scenario со структурированными эпизодами и событиями.
    """
    full_path = os.path.abspath(file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Файл сценария не найден по пути: {full_path}"
        )

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if not data:
        raise ValueError(f"Файл сценария {full_path} пуст")

    parsed_episodes = []
    raw_episodes = data.get("episodes") or []

    for ep in raw_episodes:
        # Восстанавливаем NoiseConfig, если секция есть в файле
        noise_data = ep.get("noise")
        noise_config = None
        if noise_data:
            noise_config = NoiseConfig(
                enabled=bool(noise_data.get("enabled", False)),
                file=noise_data.get("file"),
                snr_db=(
                    int(noise_data["snr_db"])
                    if noise_data.get("snr_db") is not None
                    else None
                ),
            )

        # Восстанавливаем список ScenarioEvent
        events_data = ep.get("events") or []
        events_list = [
            ScenarioEvent(
                file_id = int(ev["file_id"]),
                segment_id = int(ev["segment_id"]),
                start = float(ev["start"]),
                end = float(ev["end"]),
                text = str(ev["text"]),
                gain_db = float(ev.get("gain_db", 0.0)),
            )
            for ev in events_data
        ]

        # Собираем эпизод
        episode = ScenarioEpisode(
            id=str(ep["id"]),
            description=str(ep.get("description", "")),
            events=events_list,
            noise=noise_config,
        )
        parsed_episodes.append(episode)

    # Возвращаем финальный корневой объект сценария
    return Scenario(
        dataset_version=str(data.get("dataset_version", "0.1")),
        sample_rate=int(data.get("sample_rate", 16000)),
        episodes=parsed_episodes,
    )


def export_scenario_to_yaml(file_path: str, scenario: Scenario) -> None:
    """Выгружает объект Scenario в файл в чистом формате YAML без тегов типов.
    Args:
        file_path: Относительный или абсолютный путь для сохранения YAML-файла.
        scenario: Объект датакласса Scenario, который нужно сохранить.
    """
    full_path = os.path.abspath(file_path)

    # Создаем директорию для файла, если она еще не существует
    dir_name = os.path.dirname(full_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # dataclasses.asdict рекурсивно преобразует датаклассы в нативные dict/list
    scenario_dict = dataclasses.asdict(scenario)

    with open(full_path, "w", encoding="utf-8") as f:
        # dump параметры:
        # default_flow_style=False заставляет писать блоками (красивые списки)
        # allow_unicode=True сохраняет кириллицу в описании читаемой
        # sort_keys=False сохраняет порядок полей как в датаклассах
        yaml.dump(
            scenario_dict,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def export_cache_to_yaml(file_path: str, references: list[str], hypothesis: list[str]) -> None:
    """Сохраняет два списка строк в YAML-файл."""
    data = {"references": references, "hypothesis": hypothesis}

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def load_cache_from_yaml(file_path: str) -> tuple[list[str], list[str]] | None:
    """Загружает два списка строк из YAML-файл.
    Если файл не существует, возвращает None.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Извлекаем списки, возвращая пустые списки по умолчанию, если ключей нет
    ref_list = data.get("references", [])
    hyp_list = data.get("hypothesis", [])

    return ref_list, hyp_list
