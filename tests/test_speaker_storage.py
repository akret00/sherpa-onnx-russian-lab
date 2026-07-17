"""Модуль тестирует функционал репозитория пайплайна"""
from pathlib import Path
from typing import Generator
import numpy as np
import pytest
from entities import Speaker
from speaker_storage import BaseRepo, InMemoryRepo, SqliteRepo

# =====================================================================
# ФИКСТУРЫ (FIXTURES)
# =====================================================================

@pytest.fixture(params=["in_memory", "sqlite"])
def repo(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[BaseRepo, None, None]:
    """Параметризованная фикстура, возвращающая по очереди InMemoryRepo и SqliteRepo.
    
    Благодаря параметризации, каждый тест, использующий эту фикстуру, 
    будет запущен дважды для каждого типа репозитория.
    """
    if request.param == "in_memory":
        yield InMemoryRepo()
    elif request.param == "sqlite":
        # Используем tmp_path для создания изолированной временной БД под каждый тест
        db_file = tmp_path / "test_voice.db"
        yield SqliteRepo(db_path = str(db_file))


@pytest.fixture
def sample_embedding_v1() -> np.ndarray:
    """Возвращает тестовый вектор модели v1."""
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


@pytest.fixture
def sample_embedding_v2() -> np.ndarray:
    """Возвращает тестовый вектор модели v2."""
    return np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)


# =====================================================================
# ТЕСТЫ (TESTS)
# =====================================================================

def test_save_new_speaker_generates_id_and_name(
    repo: BaseRepo, sample_embedding_v1: np.ndarray
) -> None:
    """Проверяет, что для нового спикера генерируется ID и дефолтное имя типа SPK_001."""
    speaker = Speaker(name = None)
    speaker.add_embedding("model_v1", sample_embedding_v1)

    # До сохранения ID нет
    assert speaker.id is None
    assert speaker.embeddings[0].id is None

    # Сохраняем (объект должен мутировать)
    repo.save_speakers([speaker])

    # Проверяем генерацию ID и имени
    assert speaker.id is not None
    assert speaker.name == f"SPK_{speaker.id:03d}"
    assert speaker.embeddings[0].id is not None
    assert speaker.embeddings[0].speaker_id == speaker.id


def test_save_new_speaker_keeps_custom_name(
    repo: BaseRepo, sample_embedding_v1: np.ndarray
) -> None:
    """Проверяет, что если у спикера уже задано имя, оно не перезаписывается на SPK_00X."""
    speaker = Speaker(name="Иван")
    speaker.add_embedding("model_v1", sample_embedding_v1)

    repo.save_speakers([speaker])

    assert speaker.id is not None
    assert speaker.name == "Иван"


def test_load_speakers_without_filters_returns_all(
    repo: BaseRepo, sample_embedding_v1: np.ndarray
) -> None:
    """Проверяет, что метод load_speakers без фильтров отдает всех спикеров из БД."""
    spk1 = Speaker(name="Алла")
    spk2 = Speaker(name="Борис")
    spk1.add_embedding("model_v1", sample_embedding_v1)
    spk2.add_embedding("model_v1", sample_embedding_v1)

    repo.save_speakers([spk1, spk2])

    loaded = repo.load_speakers()
    assert len(loaded) == 2

    # Собираем имена полученных спикеров
    names = {s.name for s in loaded}
    assert names == {"Алла", "Борис"}


def test_load_speakers_filter_by_ids(repo: BaseRepo, sample_embedding_v1: np.ndarray) -> None:
    """Проверяет фильтрацию загрузки по списку ID спикеров."""
    spk1 = Speaker(name="Спикер 1")
    spk2 = Speaker(name="Спикер 2")
    spk3 = Speaker(name="Спикер 3")
    spk1.add_embedding("model_v1", sample_embedding_v1)
    spk2.add_embedding("model_v1", sample_embedding_v1)
    spk3.add_embedding("model_v1", sample_embedding_v1)

    repo.save_speakers([spk1, spk2, spk3])

    # Пытаемся загрузить только 1-го и 3-го спикера
    assert spk1.id is not None
    assert spk3.id is not None

    loaded = repo.load_speakers(speaker_ids=[spk1.id, spk3.id])

    assert len(loaded) == 2
    names = {s.name for s in loaded}
    assert names == {"Спикер 1", "Спикер 3"}


def test_load_speakers_filter_by_name(repo: BaseRepo, sample_embedding_v1: np.ndarray) -> None:
    """Проверяет фильтрацию загрузки по точному имени спикера."""
    spk1 = Speaker(name="Уникальное Имя")
    spk2 = Speaker(name="Другое Имя")
    spk1.add_embedding("model_v1", sample_embedding_v1)
    spk2.add_embedding("model_v1", sample_embedding_v1)

    repo.save_speakers([spk1, spk2])

    loaded = repo.load_speakers(name="Уникальное Имя")
    assert len(loaded) == 1
    assert loaded[0].id == spk1.id


def test_save_existing_speaker_updates_data(
    repo: BaseRepo, sample_embedding_v1: np.ndarray
) -> None:
    """Проверяет сценарий UPSERT (обновление полей у уже существующего спикера)."""
    speaker = Speaker(name="John")
    speaker.add_embedding("model_v1", sample_embedding_v1)
    repo.save_speakers([speaker])

    # Изменяем поля у объекта в памяти
    speaker.name = "John Doe"
    speaker.total_count = 150

    # Сохраняем повторно
    repo.save_speakers([speaker])

    # Выкачиваем заново из репозитория для проверки
    speaker_id = speaker.id
    assert speaker_id is not None
    loaded = repo.load_speakers(speaker_ids=[speaker_id])[0]
    assert loaded.name == "John Doe"
    assert loaded.total_count == 150


def test_multiple_embeddings_persistence(
    repo: BaseRepo,
    sample_embedding_v1: np.ndarray,
    sample_embedding_v2: np.ndarray
) -> None:
    """Проверяет, что у одного спикера успешно сохраняются и загружаются векторы разных моделей."""
    speaker = Speaker(name="MultiModel Speaker")

    # Добавляем два вектора для разных нейросетей
    speaker.add_embedding("pyannote_v3", sample_embedding_v1)
    speaker.add_embedding("respeaker_v1", sample_embedding_v2)

    repo.save_speakers([speaker])

    # Загружаем спикера обратно
    speaker_id = speaker.id
    assert speaker_id is not None
    loaded_speakers = repo.load_speakers(speaker_ids=[speaker_id])
    assert len(loaded_speakers) == 1
    loaded_speaker = loaded_speakers[0]

    # Проверяем, что оба эмбеддинга на месте и данные векторов совпадают
    assert len(loaded_speaker.embeddings) == 2

    vec_v1 = loaded_speaker.get_embedding("pyannote_v3")
    vec_v2 = loaded_speaker.get_embedding("respeaker_v1")

    assert vec_v1 is not None
    assert vec_v2 is not None
    assert np.allclose(vec_v1, sample_embedding_v1)
    assert np.allclose(vec_v2, sample_embedding_v2)


def test_in_memory_repo_data_isolation(sample_embedding_v1: np.ndarray) -> None:
    """Специфичный тест для InMemoryRepo: проверяет изоляцию данных через deepcopy.
    
    Изменение возвращенного из БД объекта в памяти не должно менять саму БД 
    до вызова метода сохранения. В реальных СУБД это базовое свойство.
    """
    repo_local = InMemoryRepo()
    speaker = Speaker(name="Original Name")
    speaker.add_embedding("model_v1", sample_embedding_v1)
    repo_local.save_speakers([speaker])

    # Шаг 1: Извлекаем из репозитория
    loaded = repo_local.load_speakers(name="Original Name")[0]

    # Шаг 2: Мутируем извлеченный объект, НО НЕ вызываем save_speakers
    loaded.name = "Mutated Name"

    # Шаг 3: Проверяем, что в репозитории объект остался прежним
    speaker_id = speaker.id
    assert speaker_id is not None
    still_original = repo_local.load_speakers(speaker_ids=[speaker_id])[0]
    assert still_original.name == "Original Name"
