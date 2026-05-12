<#
.SYNOPSIS
    Скрипт обновления файлов проекта распознавания речи.

.DESCRIPTION
    Скачивает актуальную версию проекта с GitHub, обновляет все файлы
    (кроме самого себя и win_asr_setup.bat), затем запускает скрипт
    инициализации окружения win_asr_init.ps1.

    Все временные файлы хранятся в папке tmp внутри проекта и
    гарантированно удаляются при любом исходе работы скрипта.

.EXITCODES
    0 - Успешное завершение
    1 - Нет прав на запись в текущей папке
    2 - Ошибка скачивания архива
    3 - Ошибка распаковки архива
    4 - Ошибка копирования файлов
    5 - Файл win_asr_init.ps1 отсутствует после обновления
#>

# Перехват ВСЕХ непредвиденных ошибок
trap {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  НЕПРЕДВИДЕННАЯ ОШИБКА" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Стек вызовов:" -ForegroundColor Yellow
    Write-Host $_.ScriptStackTrace -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Нажмите любую клавишу для выхода..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 99
}

# Отключаем вывод прогресса Invoke-WebRequest для более чистого лога
$ProgressPreference = 'SilentlyContinue'

# Определяем корень проекта (папка, где лежит этот скрипт)
$ProjectRoot = $PSScriptRoot

# Устанавливаем заголовок окна PowerShell
$host.UI.RawUI.WindowTitle = "Обновление распознавания речи"

Write-Host "============================================================"
Write-Host "  Обновление файлов проекта"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# Вспомогательная функция: корректное завершение с паузой
# ============================================================
function Exit-WithPause {
    param(
        [string]$Message,
        [int]$ExitCode
    )
    
    Write-Host ""
    if ($ExitCode -eq 0) {
        Write-Host $Message -ForegroundColor Green
    }
    else {
        Write-Host $Message -ForegroundColor Red
        Write-Host "Код ошибки: $ExitCode" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Нажмите любую клавишу для выхода..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit $ExitCode
}

# ============================================================
# Вспомогательная функция: очистка временной папки
# ============================================================
function Clear-TempFolder {
    $tempPath = Join-Path $ProjectRoot "tmp"
    if (Test-Path $tempPath) {
        try {
            Remove-Item -Path $tempPath -Recurse -Force -ErrorAction Stop
            Write-Host "[ОЧИСТКА] Временная папка удалена."
        }
        catch {
            Write-Host "[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить временную папку: $($_.Exception.Message)"
            Write-Host "Это не критично, папка будет перезаписана при следующем запуске."
        }
    }
}

# ============================================================
# Шаг 0. Проверка прав на запись в текущей папке
# ============================================================
Write-Host "[0/6] Проверка прав на запись..."
try {
    $testFile = Join-Path $ProjectRoot ".write_test"
    New-Item -Path $testFile -ItemType File -Force -ErrorAction Stop | Out-Null
    Remove-Item -Path $testFile -Force -ErrorAction Stop
    Write-Host "  Права на запись есть."
}
catch {
    Exit-WithPause -Message @"
[ОШИБКА] Нет прав на запись в текущую папку.

Текущая папка: $ProjectRoot

Возможные решения:
  1. Переместите папку проекта в другое место,
     например: C:\Users\$env:USERNAME\asr_project\
  2. Запустите скрипт от имени администратора
     (правый клик по win_asr_setup.bat -> Запуск от имени администратора).
"@ -ExitCode 1
}
Write-Host ""

# ============================================================
# Шаг 1. Очистка временной папки от предыдущих запусков
# ============================================================
Write-Host "[1/6] Подготовка временной папки..."
Clear-TempFolder

# Создаём временную папку заново
$tempDir = Join-Path $ProjectRoot "tmp"
try {
    New-Item -Path $tempDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
    Write-Host "  Временная папка создана: tmp\"
}
catch {
    Exit-WithPause -Message @"
[ОШИБКА] Не удалось создать временную папку.

Путь: $tempDir
Причина: $($_.Exception.Message)
"@ -ExitCode 2
}
Write-Host ""

# ============================================================
# Шаг 2. Скачивание ZIP-архива проекта с GitHub
# ============================================================
Write-Host "[2/6] Скачивание архива проекта..."
$archiveUrl = "https://github.com/akret00/sherpa-onnx-russian-lab/archive/refs/heads/main.zip"
$archivePath = Join-Path $tempDir "project_update.zip"

try {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath -ErrorAction Stop
    $archiveSize = (Get-Item $archivePath).Length
    Write-Host "  Архив скачан успешно (размер: $([math]::Round($archiveSize/1KB, 1)) КБ)"
}
catch {
    # Очищаем временную папку перед выходом
    Clear-TempFolder
    
    Exit-WithPause -Message @"
[ОШИБКА] Не удалось скачать архив проекта.

Ссылка: $archiveUrl
Причина: $($_.Exception.Message)

Проверьте:
  1. Подключение к интернету
  2. Не заблокирован ли доступ к github.com
     (попробуйте открыть ссылку в браузере)
  3. Не блокирует ли антивирус или брандмауэр скачивание
"@ -ExitCode 2
}
Write-Host ""

# ============================================================
# Шаг 3. Распаковка архива
# ============================================================
Write-Host "[3/6] Распаковка архива..."
$extractPath = Join-Path $tempDir "project_update_temp"

try {
    # Удаляем папку распаковки, если она существует (на случай повторного запуска)
    if (Test-Path $extractPath) {
        Remove-Item -Path $extractPath -Recurse -Force -ErrorAction Stop
    }
    
    Expand-Archive -Path $archivePath -DestinationPath $extractPath -Force -ErrorAction Stop
    Write-Host "  Архив распакован успешно."
}
catch {
    # Очищаем временную папку перед выходом
    Clear-TempFolder
    
    Exit-WithPause -Message @"
[ОШИБКА] Не удалось распаковать архив.

Архив: $archivePath
Причина: $($_.Exception.Message)

Возможные причины:
  1. Архив скачался с ошибкой (повреждён)
  2. Недостаточно свободного места на диске
  3. Путь содержит недопустимые символы

Попробуйте запустить скрипт повторно.
"@ -ExitCode 3
}
Write-Host ""

# ============================================================
# Шаг 4. Копирование файлов в корень проекта
# ============================================================
Write-Host "[4/6] Копирование файлов проекта..."

# Определяем папку-источник (внутри ZIP архива GitHub создаёт подпапку с именем репозитория)
$sourcePath = Join-Path $extractPath "sherpa-onnx-russian-lab-main"

if (-not (Test-Path $sourcePath)) {
    Clear-TempFolder
    Exit-WithPause -Message @"
[ОШИБКА] Не найдена папка с файлами проекта внутри архива.

Ожидаемая папка: $sourcePath

Возможно, изменилась структура репозитория на GitHub.
Свяжитесь с разработчиком для исправления скрипта.
"@ -ExitCode 4
}

try {
    # Копируем ВСЕ файлы, КРОМЕ win_asr_setup.bat и win_asr_setup.ps1
    # Это нужно, чтобы не перезаписать работающий скрипт обновления
    $excludedFiles = @("win_asr_setup.bat", "win_asr_setup.ps1")
    
    Get-ChildItem -Path $sourcePath -Force | ForEach-Object {
        if ($excludedFiles -contains $_.Name) {
            Write-Host "  Пропущен (работающий скрипт): $($_.Name)"
            return
        }
        
        $destination = Join-Path $ProjectRoot $_.Name
        try {
            Copy-Item -Path $_.FullName -Destination $destination -Recurse -Force -ErrorAction Stop
            Write-Host "  Обновлён: $($_.Name)"
        }
        catch {
            throw $_
        }
    }
    
    Write-Host "  Копирование завершено."
}
catch {
    Clear-TempFolder
    Exit-WithPause -Message @"
[ОШИБКА] Не удалось скопировать файлы проекта.

Причина: $($_.Exception.Message)

Проверьте:
  1. Достаточно ли свободного места на диске
  2. Не открыт ли какой-либо файл проекта в другой программе
  3. Есть ли права на запись в папку проекта
"@ -ExitCode 4
}
Write-Host ""

# ============================================================
# Шаг 5. Проверка наличия win_asr_init.ps1 после обновления
# ============================================================
Write-Host "[5/6] Проверка наличия скрипта инициализации..."
$initScriptPath = Join-Path $ProjectRoot "win_asr_init.ps1"

if (-not (Test-Path $initScriptPath)) {
    Clear-TempFolder
    Exit-WithPause -Message @"
[ОШИБКА] Файл win_asr_init.ps1 не найден после обновления.

Ожидаемый путь: $initScriptPath

Этот файл необходим для установки окружения (Python, ffmpeg, модели).
Возможно, он отсутствует в репозитории на GitHub.
Свяжитесь с разработчиком.
"@ -ExitCode 5
}
Write-Host "  Файл win_asr_init.ps1 найден."
Write-Host ""

# ============================================================
# Шаг 6. Очистка временной папки
# ============================================================
Write-Host "[6/6] Очистка временных файлов..."
Clear-TempFolder
Write-Host ""

# ============================================================
# Шаг 7. Запуск скрипта инициализации окружения
# ============================================================
Write-Host "============================================================"
Write-Host "  Запуск инициализации окружения"
Write-Host "============================================================"
Write-Host ""

try {
    # Запускаем скрипт в ТЕКУЩЕМ процессе PowerShell
    # Все переменные, функции и окружение останутся доступны
    & $initScriptPath
    
    # Если скрипт выполнился без ошибок (не вызвал exit с кодом ошибки)
    Exit-WithPause -Message "Обновление успешно завершено!`nТеперь можно запустить ярлык `"Распознать аудио`" на рабочем столе." -ExitCode 0
}
catch {
    # Этот catch теперь действительно работает!
    # Ловит ЛЮБЫЕ ошибки из win_asr_init.ps1
    Exit-WithPause -Message "Обновление прервано с ошибкой на этапе инициализации.`nПричина: $($_.Exception.Message)`n`nПроверьте сообщения выше. Вы можете запустить скрипт повторно." -ExitCode 6
}
