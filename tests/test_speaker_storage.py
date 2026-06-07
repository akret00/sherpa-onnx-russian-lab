"""Тест модуля src/speaker_storage.py"""
import sqlite3
import pytest
import numpy
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
    # Генерируем случайный вектор из 128 чисел типа float32
    mock_embedding1 = numpy.random.rand(128).astype(numpy.float32)
    mock_embedding2 = numpy.random.rand(128).astype(numpy.float32)
    spk1 = Speaker(name="Алексей", embedding=mock_embedding1, total_count = 11)
    spk2 = Speaker(name="Мария", embedding=mock_embedding2, total_count = 22)
    db_repo.save_speakers([spk1, spk2])
    yield spk1, spk2


# ==========================================
# ТЕСТЫ: СОХРАНЕНИЕ И АВТОИНКРЕМЕНТ
# ==========================================

def test_save_new_speakers_generates_ids(db_repo):
    """Проверяет, что у новых спикеров корректно заполняется ID из базы."""
    mock_embedding1 = numpy.random.rand(128).astype(numpy.float32)
    mock_embedding2 = numpy.random.rand(128).astype(numpy.float32)
    spk1 = Speaker(name="Unknown SPEAKER_00", embedding=mock_embedding1, total_count = 11)
    spk2 = Speaker(name="Unknown SPEAKER_01", embedding=mock_embedding2, total_count = 22)

    assert spk1.id is None
    assert spk2.id is None

    db_repo.save_speakers([spk1, spk2])

    # Проверяем мутацию объектов (ID должны проставиться)
    assert spk1.id == 1
    assert spk2.id == 2

    # Вычитываем из базы заново и проверяем правильность данных
    updated_spk1 = db_repo.load_speakers(speaker_ids=[spk1.id])[0]
    updated_spk2 = db_repo.load_speakers(speaker_ids=[spk2.id])[0]
    assert updated_spk1.id == spk1.id
    assert updated_spk1.name == "Unknown SPEAKER_00"
    assert numpy.array_equal(updated_spk1.embedding, mock_embedding1)
    assert updated_spk1.total_count == 11

    assert updated_spk2.id == spk2.id
    assert updated_spk2.name == "Unknown SPEAKER_01"
    assert numpy.array_equal(updated_spk2.embedding, mock_embedding2)
    assert updated_spk2.total_count == 22


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

def test_update_mode_all(db_repo, sample_speakers):
    """Проверяет обновление метаданных без изменения вектора."""
    spk1, _ = sample_speakers

    # Меняем имя в коде, но оставляем старый эмбеддинг
    spk1.name = "Алексей Переименованный"
    mock_embedding3 = numpy.random.rand(128).astype(numpy.float32)
    spk1.embedding = mock_embedding3 # Попытка изменить эмбеддинг
    spk1.total_count = 33

    db_repo.save_speakers([spk1], update_mode=SpeakerUpdateMode.UPDATE_ALL)

    # Вычитываем из базы заново
    updated_spk = db_repo.load_speakers(speaker_ids=[spk1.id])[0]

    assert updated_spk.name == "Алексей Переименованный"
    assert numpy.array_equal(updated_spk.embedding, mock_embedding3)
    assert updated_spk.total_count == 33


def test_update_mode_all_except_embedding(db_repo, sample_speakers):
    """Проверяет обновление метаданных без изменения вектора."""
    spk1, _ = sample_speakers

    # Меняем имя в коде, но оставляем старый эмбеддинг
    spk1.name = "Алексей Переименованный"
    old_embedding1 = spk1.embedding
    mock_embedding3 = numpy.random.rand(128).astype(numpy.float32)
    spk1.embedding = mock_embedding3 # Попытка изменить эмбеддинг
    spk1.total_count = 33

    db_repo.save_speakers([spk1], update_mode=SpeakerUpdateMode.UPDATE_ALL_EXCEPT_EMBEDDING)

    # Вычитываем из базы заново
    updated_spk = db_repo.load_speakers(speaker_ids=[spk1.id])[0]

    assert updated_spk.name == "Алексей Переименованный"
    # Эмбеддинг должен остаться СТАРЫМ
    assert numpy.array_equal(updated_spk.embedding, old_embedding1)
    assert updated_spk.total_count == 33


def test_update_mode_embeddings_only(db_repo, sample_speakers):
    """Проверяет обновление только векторов без изменения метаданных."""
    spk1, _ = sample_speakers

    spk1.name = "Хакер" # Попытка изменить имя
    mock_embedding3 = numpy.random.rand(128).astype(numpy.float32)
    spk1.embedding = mock_embedding3 # Новый эмбеддинг
    spk1.total_count = 33

    db_repo.save_speakers([spk1], update_mode=SpeakerUpdateMode.UPDATE_EMBEDDINGS_ONLY)

    # Вычитываем из базы заново
    updated_spk = db_repo.load_speakers(speaker_ids=[spk1.id])[0]

    assert updated_spk.name == "Алексей"  # Имя должно остаться СТАРЫМ
    assert numpy.array_equal(updated_spk.embedding, mock_embedding3) # Эмбеддинг обновился
    assert updated_spk.total_count == 11 # Счетчик должен остаться старым


def test_update_mode_no_update(db_repo, sample_speakers):
    """Проверяет режим полного игнорирования обновлений старых спикеров."""
    spk1, _ = sample_speakers

    spk1.name = "Новое имя"
    mock_embedding3 = numpy.random.rand(128).astype(numpy.float32)
    spk1.embedding = mock_embedding3
    spk1.total_count = 33

    # Важно: новый спикер в этом списке ДОЛЖЕН добавиться в любом случае
    mock_embedding4 = numpy.random.rand(128).astype(numpy.float32)
    new_spk = Speaker(name = "Новичок", embedding = mock_embedding4, total_count = 44)

    db_repo.save_speakers([spk1, new_spk], update_mode=SpeakerUpdateMode.NO_UPDATE)

    # Проверяем старого
    old_spk_db = db_repo.load_speakers(speaker_ids=[spk1.id])[0]

    assert old_spk_db.name == "Алексей" # Данные не изменились
    assert old_spk_db.total_count == 11 # Счетчик должен остаться старым

    # Проверяем нового
    assert new_spk.id is not None # При сохранении, объекту должен был быть выдан ИД из БД

    # Загружаем нового и проверяем
    new_spk_from_db = db_repo.load_speakers(speaker_ids=[new_spk.id])[0]
    assert new_spk_from_db.name == "Новичок"
    assert numpy.array_equal(new_spk_from_db.embedding, mock_embedding4)
    assert new_spk_from_db.total_count == 44

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
