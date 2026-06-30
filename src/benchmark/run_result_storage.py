"""Модуль занимается сохранением и загрузкой результатов экспериментов"""
# Формат названия папок:
# YYYYMMDD_HHMMSS_[Датасет]_[PipelineType]_[VAD]_[ASR]_[EMB]
# Примеры:
# 20260701_170328_single_asr_silero_gigaam3_3deres
# 20260701_170328_all_asr_oracle_gigaam3
# 20260701_170328_2dia_dcentr_oracle_oracle_3deres
# 20260701_170328_4round_dcentr_oracle_gigaam3_3deres
# Путь к папке хранения: benchmark/runs
# Содержимое папки:
# exp_spec.yaml
# pl_result.yaml
# metrics_wer.yaml
# metrics_der.yaml
from pathlib import Path
import dataclasses
from enum import Enum
from typing import Type, TypeVar
from datetime import datetime
import yaml
from config import PipelineConfig
from entities import Speaker, AudioFile, AudioSegment
from benchmark.experiment_entities import PipelineResultExperiment, ExperimentSpec

EXP_SPEC_FILE_NAME = "exp_spec.yaml"
PL_RESULT_FILE_NAME = "pl_result.yaml"

def get_result_folder_path(plres: PipelineResultExperiment) -> Path:
    """Конструирует имя папки на основе данных из результатов эксперимента"""
    time = plres.start_time.strftime("%Y%m%d_%H%M%S")
    dataset = plres.exp_spec.dataset_view
    pl_type = plres.pl_config.runtime.pipeline_type.value
    vad = plres.pl_config.vad.model_short_name
    asr = plres.pl_config.asr.model_short_name
    embed = plres.pl_config.embed.model_short_name
    folder_path = f"{time}_{dataset}_{pl_type}_{vad}_{asr}_{embed}"
    return Path(folder_path)

# --- Сериализаторы для вложенных классов PipilineResult ---
def _serialize_pl_config(pl_config: PipelineConfig) -> dict | None:
    """
    Сериализует PipelineConfig и все его вложенные плоские подконфиги.
    Автоматически обрабатывает Enum-значения.
    """
    if pl_config is None:
        return None

    pl_config_dict = {}

    # Проходим по всем полям главного конфига (runtime, vad, asr и т.д.)
    for field in dataclasses.fields(pl_config):
        sub_config = getattr(pl_config, field.name)
        if sub_config is None:
            pl_config_dict[field.name] = None
            continue

        # Так как подконфиги плоские, безопасно превращаем их в словарь
        sub_dict = dataclasses.asdict(sub_config)

        # Важно: обрабатываем Enum (например, PipelineType), переводя их в строки/значения
        for key, value in sub_dict.items():
            if isinstance(value, Enum):
                sub_dict[key] = value.value  # Берем строковое или числовое значение из Enum

        pl_config_dict[field.name] = sub_dict

    return pl_config_dict

def _serialize_audio_file(audio_file: AudioFile | None) -> dict | None:
    """
    Сериализует метаданные AudioFile.
    Исключает поле 'segments' для предотвращения циклических ссылок.
    """
    if audio_file is None:
        return None

    # Превращаем в словарь
    file_dict = dataclasses.asdict(audio_file)
    # Удаляем связь со списком сегментов
    file_dict.pop("segments", None)

    return file_dict

def _serialize_speakers(speakers: list[Speaker]) -> list[dict]:
    """
    Сериализует список объектов Speaker.
    Исключает тяжелое бинарное поле 'embedding'.
    """
    serialized_speakers = []

    for speaker in speakers:
        if speaker is None:
            continue

        spk_dict = dataclasses.asdict(speaker)
        # Удаляем тяжелый numpy-массив эмбеддингов
        spk_dict.pop("embedding", None)

        serialized_speakers.append(spk_dict)

    return serialized_speakers

def _serialize_segments(segments: list[AudioSegment] | None) -> list[dict] | None:
    """
    Сериализует список объектов AudioSegment.
    Исключает сложные вложенные объекты audio_file и speaker, 
    оставляя только их плоские ID.
    """
    if segments is None:
        return None

    serialized_segments = []

    for segment in segments:
        if segment is None:
            continue

        # Превращаем сегмент в словарь
        seg_dict = dataclasses.asdict(segment)
        # Удаляем ссылки на тяжелые/цикличные вложенные объекты
        seg_dict.pop("audio_file", None)
        seg_dict.pop("speaker", None)

        serialized_segments.append(seg_dict)

    return serialized_segments

# --- ФУНКЦИЯ ВЕРХНЕГО УРОВНЯ ---
def export_pipeline_result_to_yaml(
    file_path: Path | str,
    pl_result: PipelineResultExperiment
) -> None:
    """
    Экспортирует объект PipelineResultExperiment (без спецификации эксперимента) в YAML файл.
    """
    file_path = Path(file_path)
    # Собираем словарь верхнего уровня вручную
    result_dict = {
        # 1. Метаданные запуска
        "start_time": pl_result.start_time.isoformat,
        "proc_time": pl_result.proc_time,
        "total_ram_mb": pl_result.total_ram,
        "sherpa_version": pl_result.sherpa_version,

        # 2. Конфигурация пайплайна
        "pl_config": _serialize_pl_config(pl_result.pl_config),

        # 3. Информация о файле
        "file": _serialize_audio_file(pl_result.file),

        # 4. Список спикеров
        "speakers": _serialize_speakers(pl_result.speakers),

        # 5. Сегменты (Гипотеза и Эталон)
        "hypo_segments": _serialize_segments(pl_result.segments),
        "markup_segments": _serialize_segments(pl_result.markup_segments)
    }

    # Сохраняем в YAML безопасным методом
    file_path.parent.mkdir(parents = True, exist_ok = True)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            result_dict,
            f,
            allow_unicode=True,  # Чтобы русский текст не превращался в \u0430
            sort_keys=False      # Сохраняем порядок полей как в словаре для читаемости
        )

T = TypeVar("T")

def _safe_instantiate(cls: Type[T], data: dict | None) -> T | None:
    """
    Безопасно создает экземпляр датакласса cls из словаря data.
    - Игнорирует лишние ключи.
    - Подставляет дефолтные значения для отсутствующих.
    - Автоматически восстанавливает Enum из строк/значений.
    - Приводит базовые типы (int, float, bool, str) к типам подсказок (Type Hints).
    """
    if data is None or not isinstance(data, dict):
        return None

    valid_fields = {f.name: f for f in dataclasses.fields(cls)}
    init_kwargs = {}

    for key, value in data.items():
        if key not in valid_fields:
            continue

        field_def = valid_fields[key]
        field_type = field_def.type

        # 1. Рекурсивная обработка вложенных датаклассов
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            init_kwargs[key] = _safe_instantiate(field_type, value)
            continue

        # 2. Восстановление Enum (например, PipelineType)
        # Проверяем, является ли тип (или его базовая часть) подклассом Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            try:
                # Пытаемся восстановить по значению (value), например:
                # "asr" -> PipelineType.ASR_PIPELINE
                init_kwargs[key] = field_type(value)
            except ValueError:
                try:
                    # Если сохранено было имя (name), например: "ASR_PIPELINE"
                    init_kwargs[key] = field_type[value]
                except KeyError:
                    # Если в старом файле неизвестный или удаленный тип, ставим None или дефолт
                    init_kwargs[key] = None
            continue

        # 3. Защита от несоответствия базовых типов (Type Cast)
        if value is not None:
            # Обработка простых типов int, float, bool, str
            if field_type in (int, float, bool, str):
                try:
                    if field_type is bool and isinstance(value, str):
                        init_kwargs[key] = value.lower() in ("true", "1", "yes")
                    else:
                        init_kwargs[key] = field_type(value)
                except (ValueError, TypeError):
                    # Если конвертация провалилась (например, "текст" -> int),
                    # мы НЕ пишем битое значение, а позволяем Python взять дефолт
                    continue
            else:
                init_kwargs[key] = value
        else:
            # Если в YAML пришел явный null, но у поля в датаклассе есть дефолт отличное от None,
            # лучше пропустить его, чтобы применился безопасный дефолт из кода.
            if (
                field_def.default is not dataclasses.MISSING
                or field_def.default_factory is not dataclasses.MISSING
            ):
                continue
            init_kwargs[key] = value

    return cls(**init_kwargs)

def load_pipeline_result_from_yaml(file_path: Path | str) -> PipelineResultExperiment | None:
    """
    Загружает и восстанавливает объект PipelineResultExperiment из YAML.
    Устойчив к изменениям в структуре классов (удаление/добавление полей).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        # safe_load возвращает чистые питоновские dict/list/базовые типы
        raw_data = yaml.safe_load(f)

    if not raw_data or not isinstance(raw_data, dict):
        return None

    # 1. Восстанавливаем плоские метаданные верхнего уровня
    start_time_raw = raw_data.get("start_time")
    start_time = None
    if start_time_raw:
        try:
            start_time = datetime.fromisoformat(start_time_raw)
        except ValueError:
            pass  # Если формат даты сломался, оставляем None

    # 2. Восстанавливаем конфигурацию PipelineConfig и подконфиги
    # Примечание: Строковые значения Enum (например, pipeline_type) запишутся как строки.
    # Если в коде вам строго нужен Enum объект, обновите логику в рантайме или
    # добавьте сюда валидатор.
    pl_config = _safe_instantiate(PipelineConfig, raw_data.get("pl_config"))

    # 3. Восстанавливаем AudioFile
    audio_file = _safe_instantiate(AudioFile, raw_data.get("file"))

    # 4. Восстанавливаем список спикеров
    speakers = []
    raw_speakers = raw_data.get("speakers", []) or []
    for spk_dict in raw_speakers:
        spk_obj = _safe_instantiate(Speaker, spk_dict)
        if spk_obj:
            speakers.append(spk_obj)

    # 5. Восстанавливаем списки сегментов (гипотеза и эталон)
    def restore_segments(raw_list) -> list[AudioSegment] | None:
        if raw_list is None:
            return None
        segments = []
        for seg_dict in raw_list:
            seg_obj = _safe_instantiate(AudioSegment, seg_dict)
            if seg_obj:
                segments.append(seg_obj)
        return segments

    segments = restore_segments(raw_data.get("segments"))
    markup_segments = restore_segments(raw_data.get("markup_segments"))

    # 6. Собираем финальный PipelineResultExperiment
    return PipelineResultExperiment(
        pl_config=pl_config,
        speakers=speakers,
        file=audio_file,
        segments=segments,
        markup_segments=markup_segments,
        start_time=start_time,
        proc_time=raw_data.get("proc_time"),
        total_ram=raw_data.get("total_ram_mb"),  # Маппим старое имя ключа в поле класса
        sherpa_version=raw_data.get("sherpa_version")
    )

def export_experiment_spec_to_yaml(file_path: Path, exp_spec: ExperimentSpec) -> None:
    """
    Экспортирует объект ExperimentSpec в YAML-файл.
    Корректно обрабатывает типы Path и Enum.
    """
    # Превращаем в плоский словарь «в лоб»
    spec_dict = dataclasses.asdict(exp_spec)

    # Обрабатываем специфичные типы, которые YAML не умеет писать красиво из коробки
    for key, value in spec_dict.items():
        # Конвертируем Path в обычные строки
        if isinstance(value, Path):
            spec_dict[key] = str(value)
        # Конвертируем Enum в строковые значения
        elif isinstance(value, Enum):
            spec_dict[key] = value.value

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            spec_dict,
            f,
            allow_unicode=True,
            sort_keys=False
        )

def load_experiment_spec_from_yaml(file_path: Path) -> ExperimentSpec | None:
    """
    Загружает и восстанавливает объект ExperimentSpec из YAML.
    Устойчив к любым изменениям структуры полей в классе ExperimentSpec.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not raw_data or not isinstance(raw_data, dict):
        return None

    # Пользуемся нашей умной функцией _safe_instantiate, которая:
    # 1. Проигнорирует удаленные поля
    # 2. Подставит дефолты из кода для новых полей
    # 3. Автоматически восстановит PipelineType (Enum) из строки
    spec_obj = _safe_instantiate(ExperimentSpec, raw_data)

    if spec_obj is None:
        return None

    # Пост-обработка: восстанавливаем типы Path, так как _safe_instantiate
    # воспринял их как обычные строки (field_type у них Path, а не str/int/float)
    if spec_obj.audio_path and not isinstance(spec_obj.audio_path, Path):
        spec_obj.audio_path = Path(spec_obj.audio_path)

    if spec_obj.ground_truth_path and not isinstance(spec_obj.ground_truth_path, Path):
        spec_obj.ground_truth_path = Path(spec_obj.ground_truth_path)

    return spec_obj

def export_plres_exp_to_yaml(plres: PipelineResultExperiment) -> None:
    """Сохраняет результат экспериментов в yaml файлы"""
    # Конструирование имени папки
    folder_path = get_result_folder_path(plres = plres)
    # Экспорт спецификации эксперимента
    export_experiment_spec_to_yaml(
        file_path = folder_path / EXP_SPEC_FILE_NAME,
        exp_spec = plres.exp_spec
    )
    # Экспорт результатов эксперимента
    export_pipeline_result_to_yaml(
        file_path = folder_path / PL_RESULT_FILE_NAME,
        pl_result = plres
    )

def load_plres_exp_from_yaml(folder_path: Path) -> PipelineResultExperiment:
    """Загружает результат экспериментов из yaml файлов"""
    # Импорт спецификации эксперимента
    plres = load_pipeline_result_from_yaml(file_path = folder_path / PL_RESULT_FILE_NAME)
    plres.exp_spec = load_experiment_spec_from_yaml(file_path = folder_path / EXP_SPEC_FILE_NAME)
    return plres
