"""Тест модуля src/speaker_storage.py"""
import sqlite3
import pytest
from speaker_storage import (
    VoiceDbRepository,
    Speaker,
    AudioFile,
    AudioSegment,
    SpeakerUpdateMode
)

# ==========================================
# ФИКСТУРЫ (Настройка окружения для каждого теста)
# ==========================================

@pytest.fixture
def db_repo(tmp_path):
    """Фикстура, которая возвращает объект VoiceDbRepository"""
    repo = VoiceDbRepository(db_path=str(tmp_path / "test_voice.db"))

    yield repo
    # Здесь может быть код очистки или завершения, когда понадобится


@pytest.fixture
def sample_speakers(db_repo):
    """Наполняет базу базовыми спикерами для тестов чтения и обновлений."""
    spk1 = Speaker(name="Алексей", embedding=b"\x01\x02")
    spk2 = Speaker(name="Мария", embedding=b"\x03\x04")
    db_repo.save_speakers([spk1, spk2])
    yield spk1, spk2


# ==========================================
# ТЕСТЫ: СОХРАНЕНИЕ И АВТОИНКРЕМЕНТ
# ==========================================

def test_save_new_speakers_generates_ids(db_repo):
    """Проверяет, что у новых спикеров корректно заполняется ID из базы."""
    spk1 = Speaker(name="Unknown SPEAKER_00", embedding=b"\x11\x22")
    spk2 = Speaker(name="Unknown SPEAKER_01", embedding=b"\x33\x44")

    assert spk1.id is None
    assert spk2.id is None

    db_repo.save_speakers([spk1, spk2])

    # Проверяем мутацию объектов (ID должны проставиться)
    assert spk1.id == 1
    assert spk2.id == 2


def test_save_audio_file_and_segments(db_repo, sample_speakers):
    """Проверяет сквозной процесс сохранения карточки файла и его сегментов."""
    spk1, spk2 = sample_speakers

    # 1. Создаем и сохраняем файл
    audio = AudioFile(file_path="/audio/test.wav", duration_seconds=60.0)
    file_id = db_repo.save_audio_file(audio)

    assert file_id is not None
    assert audio.id == file_id

    # 2. Создаем сегменты диаризации
    segments = [
        AudioSegment(
            speaker_id=spk1.id,
            start_time=0.0,
            end_time=5.5,
            text="Привет",
            word_count=1
        ),
        AudioSegment(
            speaker_id=spk2.id,
            start_time=6.0,
            end_time=12.2,
            text="Добрый день",
            word_count=2
        )
    ]
    db_repo.save_audio_segments(audio_file_id=file_id, segments=segments)

    # 3. Вычитываем обратно и проверяем целостность данных
    loaded_segments, loaded_speakers = db_repo.load_file_content(audio_file_id=file_id)

    assert len(loaded_segments) == 2
    assert len(loaded_speakers) == 2

    assert loaded_segments[0].text == "Привет"
    assert loaded_segments[0].speaker_id == spk1.id
    assert loaded_segments[1].text == "Добрый день"
    assert loaded_segments[1].speaker_id == spk2.id


# ==========================================
# ТЕСТЫ: РЕЖИМЫ ОБНОВЛЕНИЯ СПИКЕРОВ (UPDATE MODES)
# ==========================================

def test_update_mode_all_except_embedding(db_repo, sample_speakers):
    """Проверяет обновление метаданных без изменения вектора."""
    spk1, _ = sample_speakers

    # Меняем имя в коде, но оставляем старый эмбеддинг
    spk1.name = "Алексей Переименованный"
    spk1.embedding = b"\x99\x99" # Попытка изменить эмбеддинг

    db_repo.save_speakers([spk1], mode=SpeakerUpdateMode.UPDATE_ALL_EXCEPT_EMBEDDING)

    # Вычитываем из базы заново
    updated_spk = db_repo.load_speakers(speaker_ids=[spk1.id])[0]
    assert updated_spk.name == "Алексей Переименованный"
    assert updated_spk.embedding == b"\x01\x02"  # Эмбеддинг должен остаться СТАРЫМ


def test_update_mode_embeddings_only(db_repo, sample_speakers):
    """Проверяет обновление только векторов без изменения метаданных."""
    spk1, _ = sample_speakers

    spk1.name = "Хакер" # Попытка изменить имя
    spk1.embedding = b"\xFF\xFF" # Новый эмбеддинг

    db_repo.save_speakers([spk1], mode=SpeakerUpdateMode.UPDATE_EMBEDDINGS_ONLY)

    # Вычитываем из базы заново
    updated_spk = db_repo.load_speakers(speaker_ids=[spk1.id])[0]
    assert updated_spk.name == "Алексей"  # Имя должно остаться СТАРЫМ
    assert updated_spk.embedding == b"\xFF\xFF"  # Эмбеддинг обновился


def test_update_mode_no_update(db_repo, sample_speakers):
    """Проверяет режим полного игнорирования обновлений старых спикеров."""
    spk1, _ = sample_speakers

    spk1.name = "Новое имя"
    spk1.embedding = b"\x00\x00"

    # Важно: новый спикер в этом списке ДОЛЖЕН добавиться в любом случае
    new_spk = Speaker(name="Новичок", embedding=b"\x55")

    db_repo.save_speakers([spk1, new_spk], mode=SpeakerUpdateMode.NO_UPDATE)

    # Проверяем старого
    old_spk_db = db_repo.load_speakers(speaker_ids=[spk1.id])[0]
    assert old_spk_db.name == "Алексей" # Данные не изменились

    # Проверяем нового
    assert new_spk.id is not None
    assert len(db_repo.load_speakers()) == 3


# ==========================================
# ТЕСТЫ: ФИЛЬТРАЦИЯ И ПОИСК (READ POOL)
# ==========================================

def test_load_speakers_with_filters(db_repo, sample_speakers):
    """Проверяет работу фильтрации по ID и части имени LIKE."""
    spk1, spk2 = sample_speakers

    # Поиск по списку ID
    res_ids = db_repo.load_speakers(speaker_ids=[spk2.id])
    assert len(res_ids) == 1
    assert res_ids[0].name == "Мария"

    # Поиск по подстроке имени (регистронезависимый LIKE в SQLite для ASCII)
    res_name = db_repo.load_speakers(name="лекс")
    assert len(res_name) == 1
    assert res_name[0].id == spk1.id


def test_load_audio_file_by_filters(db_repo):
    """Проверяет поиск аудиофайла по ID и уникальному пути."""
    path = "/var/audio/test_file.mp3"
    audio = AudioFile(file_path=path, duration_seconds=12.5)
    db_repo.save_audio_file(audio)

    # Ищем по пути
    file_by_path = db_repo.load_audio_file(file_path=path)
    assert file_by_path is not None
    assert file_by_path.id == audio.id

    # Ищем по ID
    file_by_id = db_repo.load_audio_file(file_id=audio.id)
    assert file_by_id.file_path == path


# ==========================================
# ТЕСТЫ: ЦЕЛОСТНОСТЬ И ОГРАНИЧЕНИЯ (CONSTRAINTS)
# ==========================================

def test_audio_file_path_uniqueness(db_repo):
    """Проверяет, что база вызовет ошибку при попытке дублировать путь к файлу."""
    audio1 = AudioFile(file_path="/dup/path.wav", duration_seconds=10)
    audio2 = AudioFile(file_path="/dup/path.wav", duration_seconds=20)

    db_repo.save_audio_file(audio1)

    with pytest.raises(sqlite3.IntegrityError):
        db_repo.save_audio_file(audio2)
