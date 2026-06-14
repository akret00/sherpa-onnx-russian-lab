"""Модуль содержит функции выгрузки и загрузки результатов разметки в YAML формат"""
from pathlib import Path
import yaml
from entities import Speaker, AudioFile, AudioSegment


def export_to_yaml(
    output_path: str | Path,
    # speakers: list[Speaker],
    file: AudioFile,
) -> None:
    """Экспорт датаклассов в чистый человекочитаемый YAML."""
    spk_id = "SPK01"
    file_id = "F01"

    # 1. Формируем секцию speakers (убираем эмбеддинги и счетчики)
    yaml_speakers = []
    # for spk in speakers:
    #     yaml_speakers.append({"id": spk.id, "name": spk.name, "notes": "laptop mic"})
    yaml_speakers.append({"id": spk_id, "name": spk_id, "notes": "laptop mic"})

    # 2. Формируем секцию files
    yaml_files = []
    # yaml_files.append({"id": file.id, "file_path": file.file_path})
    yaml_files.append({"id": file_id, "file_path": file.file_path})

    # 3. Формируем секцию segments, собирая их из всех файлов
    yaml_segments = []
    if file.segments:
        seg_id = 0
        for seg in file.segments:
            seg_id += 1
            yaml_segments.append(
                {
                    "id": f"{spk_id}_{file_id}_{seg_id:04d}",
                    "speaker_id": spk_id,
                    "file_id": file_id,
                    "text": seg.text,
                    "speech_start": round(seg.start_time, 2),
                    "speech_end": round(seg.end_time, 2),
                }
            )

    # Собираем финальный документ
    data_to_save = {
        "dataset_version": "0.1",
        "sample_rate": "16000",
        "speakers": yaml_speakers,
        "files": yaml_files,
        "segments": yaml_segments,
    }

    # Сохраняем с отключением дефолтных python-тегов и красивыми отступами
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data_to_save,
            f,
            allow_unicode = True,
            sort_keys = False,
            default_flow_style = False,
        )


def load_from_yaml(yaml_path: str | Path) -> tuple[list[Speaker], list[AudioFile]]:
    """Загрузка из YAML и восстановление связей в датаклассах."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 1. Восстанавливаем спикеров
    speakers_dict = {}
    for spk_data in data.get("speakers", []):
        spk = Speaker(id=spk_data["id"], name=spk_data["name"])
        speakers_dict[spk.id] = spk

    # 2. Восстанавливаем файлы
    files_dict = {}
    for file_data in data.get("files", []):
        audio_file = AudioFile(
            id=file_data["id"], file_path=file_data["file_path"], segments=[]
        )
        files_dict[audio_file.id] = audio_file

    # 3. Восстанавливаем сегменты и связываем их
    for seg_data in data.get("segments", []):
        f_id = seg_data["file_id"]
        spk_id = seg_data["speaker_id"]

        audio_file = files_dict.get(f_id)
        speaker = speakers_dict.get(spk_id)

        segment = AudioSegment(
            id=seg_data["id"],
            audio_file_id=f_id,
            audio_file=audio_file,  # циклическая ссылка обратно на файл
            speaker_id=spk_id,
            speaker=speaker,  # ссылка на объект спикера
            start_time=float(seg_data["speech_start"]),
            end_time=float(seg_data["speech_end"]),
            text=seg_data["text"],
            word_count=len(seg_data["text"].split())
            if seg_data["text"]
            else 0,
        )

        # Добавляем сегмент в список родительского файла
        if audio_file and audio_file.segments is not None:
            audio_file.segments.append(segment)

    return list(speakers_dict.values()), list(files_dict.values())
