"""
Менеджер голосовых профилей (CLI)
Управление спикерами: просмотр, переименование, сравнение, удаление
"""
import os
import platform
import math
import numpy
import speaker_storage

# --------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------
MAX_DATA_ROWS = 10
FORM_COLS = 2
PAGE_SIZE = MAX_DATA_ROWS * FORM_COLS # Сколько спикеров на одной странице
COL_WIDTH_ID = 3
COL_WIDTH_NAME = 15
COL_WIDTH_SEGMENTS = 10
COLS_DELIMITER = " | "
FORM_WIDTH = (
    (COL_WIDTH_ID + COL_WIDTH_NAME + COL_WIDTH_SEGMENTS +1) * FORM_COLS
    + len(COLS_DELIMITER) * (FORM_COLS - 1)
)
FILL_MODE_HORISINTAL = "horisontal"
FILL_MODE_VERTICAL = "vertical"
FILL_MODE = FILL_MODE_VERTICAL
# FILL_MODE = FILL_MODE_HORISINTAL

# --------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------
def clear_screen():
    """Очищает экран консоли (Linux/Windows)."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def cosine_similarity(emb1: numpy.ndarray, emb2: numpy.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя эмбеддингами."""
    if emb1 is None or emb2 is None:
        return 0.0
    return numpy.dot(emb1, emb2) # Косинусное для нормированных векторов

def press_enter_to_continue():
    """Ждет нажатия Enter."""
    input("\nНажмите Enter для продолжения...")

def get_int_input(prompt: str, min_val: int = 1, max_val: int | None = None) -> int | None:
    """
    Безопасный ввод целого числа.
    Возвращает None, если ввод пустой (отмена).
    """
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            val = int(raw)
            if val < min_val or (max_val is not None and val > max_val):
                print(f"Пожалуйста, введите число от {min_val}"
                      + (f" до {max_val}" if max_val else ""))
                continue
            return val
        except ValueError:
            print("Пожалуйста, введите целое число.")


# --------------------------------------------------------------
# Операции со спикерами
# --------------------------------------------------------------

def rename_speaker(repo: VoiceDbRepository) -> bool:
    """Переименование спикера. Возвращает True, если что-то изменилось."""
    sp_id = get_int_input("Введите ID спикера для переименования (Enter для отмены): ")
    if sp_id is None:
        return False

    # Загружаем одного спикера
    speakers = repo.load_speakers(speaker_ids=[sp_id])
    if not speakers:
        print(f"Спикер с ID={sp_id} не найден.")
        return False

    sp = speakers[0]
    print(f"\nТекущее имя: {sp.name}")
    new_name = input("Новое имя: ").strip()

    if not new_name:
        print("Переименование отменено (пустое имя).")
        return False

    if new_name == sp.name:
        print("Имя не изменилось.")
        return False

    # Сохраняем с новым именем
    sp.name = new_name
    repo.save_speakers(speakers = [sp], update_mode = SpeakerUpdateMode.UPDATE_ALL_EXCEPT_EMBEDDING)
    print(f"Спикер ID={sp.id} переименован в '{new_name}'")
    return True


def compare_speakers(repo: VoiceDbRepository):
    """Сравнение двух спикеров по косинусному сходству эмбеддингов."""
    print("\nСравнение двух спикеров по сходству голосов:")

    id1 = get_int_input("Введите ID первого спикера (Enter для отмены): ")
    if id1 is None:
        return

    id2 = get_int_input("Введите ID второго спикера (Enter для отмены): ")
    if id2 is None:
        return

    if id1 == id2:
        print("Вы выбрали одного и того же спикера. Сходство всегда 100%.")
        return

    # Загружаем обоих
    speakers = repo.load_speakers(speaker_ids=[id1, id2])
    sp_dict = {sp.id: sp for sp in speakers}

    if id1 not in sp_dict:
        print(f"Спикер с ID={id1} не найден.")
        return
    if id2 not in sp_dict:
        print(f"Спикер с ID={id2} не найден.")
        return

    sp1, sp2 = sp_dict[id1], sp_dict[id2]

    # Проверяем наличие эмбеддингов
    if sp1.embedding is None or sp2.embedding is None:
        print("У одного из спикеров отсутствует эмбеддинг (голосовой образец).")
        return

    similarity = cosine_similarity(sp1.embedding, sp2.embedding)
    similarity_pct = similarity * 100

    # Интерпретация
    if similarity_pct > 85:
        interpretation = "Очень высокая — вероятно, один человек"
    elif similarity_pct > FORM_WIDTH:
        interpretation = "Высокая — возможно, один человек"
    elif similarity_pct > 50:
        interpretation = "Средняя — требуется дополнительная проверка"
    else:
        interpretation = "Низкая — вероятно, разные люди"

    print(f"\n{'=' * 50}")
    print(f"  Спикер 1: [{sp1.id}] {sp1.name}")
    print(f"  Спикер 2: [{sp2.id}] {sp2.name}")
    print(f"  Косинусное сходство: {similarity:.4f} ({similarity_pct:.1f}%)")
    print(f"  Интерпретация: {interpretation}")
    print(f"{'=' * 50}")


def delete_speaker(repo: VoiceDbRepository) -> bool:
    """
    Удаление спикера с обязательным подтверждением.
    Возвращает True, если удаление выполнено.
    """
    sp_id = get_int_input("Введите ID спикера для удаления (Enter для отмены): ")
    if sp_id is None:
        return False

    speakers = repo.load_speakers(speaker_ids=[sp_id])
    if not speakers:
        print(f"Спикер с ID={sp_id} не найден.")
        return False

    sp = speakers[0]

    # Предупреждение, если есть записи
    print("\nВНИМАНИЕ! Вы собираетесь удалить спикера:")
    print(f"   ID: {sp.id}")
    print(f"   Имя: {sp.name}")
    print(f"   Количество фраз: {sp.total_count}")

    if sp.total_count > 0:
        print(f"\n   У этого спикера есть {sp.total_count} привязанных фраз.")
        print("   После удаления эти фразы останутся без привязки к спикеру.")

    # Двойное подтверждение
    confirm = input("\nДля подтверждения удаления введите 'удалить': ").strip().lower()
    if confirm != "удалить":
        print("Удаление отменено.")
        return False

    # Тут нужен метод delete в твоем репозитории
    # Если его нет — добавь или используй прямой SQL
    try:
        # Временное решение, пока нет метода delete:
        # repo.delete_speaker(sp_id)
        # Вместо этого можно пометить как удаленного:
        sp.name = f"_DELETED_{sp.name}"
        repo.save_speakers([sp])
        print(f"Спикер ID={sp.id} помечен как удаленный.")
        print("  (Добавьте метод delete_speaker в VoiceDbRepository для физического удаления)")
    except RuntimeError as e:
        print(f"Ошибка при удалении спикера: {e}")
        return False

    return True

# --------------------------------------------------------------
# Форматирование одноо спикера
# --------------------------------------------------------------
def format_speaker(sp) -> str:
    """Форматирует одного спикера в одну группу колонок"""
    if not sp:
        # Возвращаем пустые колонки, если данных для этого места нет
        return f" {' ':<{COL_WIDTH_ID}} {' ':<{COL_WIDTH_NAME}} {' ':<{COL_WIDTH_SEGMENTS}}"

    name = sp.name if sp.name else "Без имени"
    if len(name) > COL_WIDTH_NAME:
        display_name = name[:COL_WIDTH_NAME - 2] + ".."
    else:
        display_name = name

    return (
        f" {sp.id:<{COL_WIDTH_ID}} "
        f"{display_name:<{COL_WIDTH_NAME}} "
        f"{sp.total_count:<{COL_WIDTH_SEGMENTS}}"
    )

# --------------------------------------------------------------
# Заголовок формы
# --------------------------------------------------------------
def show_header():
    """Показывает заголовок формы"""
    # print(f"\n{'═' * FORM_WIDTH}")
    print("  МЕНЕДЖЕР ГОЛОСОВЫХ ПРОФИЛЕЙ")
    print(f"\n{'═' * FORM_WIDTH}")

# --------------------------------------------------------------
# Отображение данных
# --------------------------------------------------------------
def print_speakers_table(speakers: list[Speaker], page: int = 1, total_count: int = 0):
    """
    Выводит таблицу спикеров с пагинацией.
    
    Args:
        speakers: список спикеров для отображения (уже отфильтрованный)
        page: текущая страница
        total_count: общее количество спикеров (для пагинации)
    """
    if not speakers:
        print("\n   Спикеры не найдены.")
        return

    # Заголовок
    # print(f"\n{'-' * FORM_WIDTH}")
    col_title = (
        f" {'ID':<{COL_WIDTH_ID}} "
        f"{'Имя':<{COL_WIDTH_NAME}} "
        f"{'Фразы':<{COL_WIDTH_SEGMENTS}}"
    )
    # Повторяет заголовок по количеству колонок формы FORM_COLS
    print(COLS_DELIMITER.join(col_title for _ in range(FORM_COLS)))
    print(f"{'-' * FORM_WIDTH}")

    # Данные
    if FILL_MODE == FILL_MODE_VERTICAL:
        # --- РЕЖИМ 1: Сверху вниз, затем следующая колонка (По вертикали) ---
        # Рассчитываем, сколько строк реально понадобится
        # (но не больше лимита для заполнения колонок)
        total_needed_rows = math.ceil(len(speakers) / FORM_COLS)
        rows_count = min(MAX_DATA_ROWS, total_needed_rows)

        for row_idx in range(rows_count):
            row_cells = []
            for col_idx in range(FORM_COLS):
                # Вычисляем индекс элемента в исходном списке для текущей строки и колонки
                element_idx = col_idx * rows_count + row_idx

                if element_idx < len(speakers):
                    row_cells.append(speakers[element_idx])
                else:
                    row_cells.append(None)

            print(COLS_DELIMITER.join(format_speaker(sp) for sp in row_cells))
    else:
        # --- РЕЖИМ 2: По очереди во все колонки (Слева направо, сверху вниз) ---
        for i in range(0, len(speakers), FORM_COLS):
            # Берем срез элементов для одной строки таблицы
            chunk = speakers[i:i + FORM_COLS]
            # Если элементов меньше, чем колонок, дополняем None
            while len(chunk) < FORM_COLS:
                chunk.append(None)

            # Форматируем каждый элемент и соединяем их в одну строку
            print(COLS_DELIMITER.join(format_speaker(sp) for sp in chunk))

    print(f"{'-' * FORM_WIDTH}")

    # Информация о странице
    if total_count > PAGE_SIZE:
        total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
        print(f"  Страница {page}/{total_pages} (всего спикеров: {total_count})")
    else:
        print(f"  Всего спикеров: {len(speakers)}")


# --------------------------------------------------------------
# Главное меню
# --------------------------------------------------------------

def show_menu(search_query: str = ""):
    """Показывает главное меню."""
    print(f"{'═' * FORM_WIDTH}")
    print("  [P] Предыдущая страница  [N] Следующая страница")
    print("  [S] Поиск по имени       [A] Показать всех")
    print(f"{'-' * FORM_WIDTH}")
    print("  [R] Переименовать спикера")
    print("  [C] Сравнить двух спикеров")
    print("  [D] Удалить спикера")
    print(f"{'-' * FORM_WIDTH}")
    print("  [Q] Выход")
    print(f"{'═' * FORM_WIDTH}")

    if search_query:
        print(f"  Поиск: '{search_query}'")


def main():
    """Главный цикл приложения."""
    repo = VoiceDbRepository()

    page = 1
    search_query = ""

    while True:
        clear_screen()

        # Загружаем спикеров с учетом поиска
        if search_query:
            speakers = repo.load_speakers(name=search_query)
        else:
            speakers = repo.load_speakers()

        total_count = len(speakers)
        total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)

        # Корректируем страницу
        page = min(max(page, 1), total_pages)

        # Нарезаем страницу
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_speakers = speakers[start_idx:end_idx]

        # Показываем заголовок
        show_header()
        # Показываем таблицу
        print_speakers_table(page_speakers, page, total_count)
        # Показываем меню
        show_menu(search_query)

        # Обработка ввода
        choice = input("\nВаш выбор: ").strip().lower()

        if choice == 'q':
            print("До свидания!")
            break

        if choice == 'p':
            if page > 1:
                page -= 1

        elif choice == 'n':
            if page < total_pages:
                page += 1

        elif choice == 's':
            search_query = input("Введите имя или часть имени для поиска: ").strip()
            page = 1  # Сбрасываем страницу при новом поиске
            if not search_query:
                search_query = ""  # Показываем всех

        elif choice == 'a':
            search_query = ""
            page = 1

        elif choice == 'r':
            if rename_speaker(repo):
                press_enter_to_continue()

        elif choice == 'c':
            compare_speakers(repo)
            press_enter_to_continue()

        elif choice == 'd':
            if delete_speaker(repo):
                press_enter_to_continue()

        elif choice.isdigit():
            # Прямой переход на страницу
            target_page = int(choice)
            if 1 <= target_page <= total_pages:
                page = target_page
            else:
                print(f"Нет страницы {target_page}. Доступны страницы 1-{total_pages}")
                press_enter_to_continue()

        else:
            print("Неизвестная команда. Используйте буквы из меню.")
            press_enter_to_continue()


# --------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------
if __name__ == "__main__":
    main()
