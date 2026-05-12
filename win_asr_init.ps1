<#
.SYNOPSIS
    Скрипт инициализации окружения для распознавания речи.

.DESCRIPTION
    Устанавливает всё необходимое для работы проекта в портабельном режиме
    (без прав администратора):
    - Проверяет конфигурационный файл
    - Проверяет наличие Visual C++ Redistributable
    - Устанавливает портативный Python 3.14.4 в bin\python\
    - Устанавливает pip и зависимости из requirements.txt
    - Скачивает ffmpeg в bin\ffmpeg.exe
    - Скачивает ASR-модель GigaAM v3
    - Скачивает VAD-модель Silero

.NOTES
    Все временные файлы хранятся в папке tmp внутри проекта.
    Папка tmp гарантированно удаляется при любом исходе.

.EXITCODES
    0 - Успешное завершение
    1 - Отсутствует config.yaml.sample
    2 - Ошибка скачивания Python
    3 - Ошибка установки pip
    4 - Отсутствует requirements.txt или ошибка установки пакетов
    5 - Ошибка установки ffmpeg
    6 - Ошибка загрузки ASR-модели
    7 - Ошибка загрузки VAD-модели
    9 - Требуется установка Visual C++ Redistributable
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

# Отключаем вывод прогресса Invoke-WebRequest для чистого лога
$ProgressPreference = 'SilentlyContinue'

# Определяем корень проекта (папка, где лежит этот скрипт)
$ProjectRoot = $PSScriptRoot

# Python
$PythonDir = Join-Path $ProjectRoot "bin\python"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonVersion = "3.14.4"
$PythonZip = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonZip"

# ffmpeg
$FfmpegDir = Join-Path $ProjectRoot "bin"
$FfmpegExe = Join-Path $FfmpegDir "ffmpeg.exe"
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

# Модели
$AsrModelDir = Join-Path $ProjectRoot "models\asr\giga-am-v3"
$AsrModelFile1 = Join-Path $AsrModelDir "model.int8.onnx"
$AsrModelFile2 = Join-Path $AsrModelDir "tokens.txt"
$AsrModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16.tar.bz2"

$VadModelFile = Join-Path $ProjectRoot "models\vad\silero\silero_vad.onnx"
$VadModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

# Временная папка
$TempDir = Join-Path $ProjectRoot "tmp"

# Устанавливаем заголовок окна PowerShell
$host.UI.RawUI.WindowTitle = "Инициализация окружения распознавания речи"

# ============================================================
# Вспомогательные функции
# ============================================================

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  $Message"
    Write-Host ("=" * 60)
}

function Write-Success {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "  [ВНИМАНИЕ] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "  [ОШИБКА] $Message" -ForegroundColor Red
}

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

function Clear-TempFolder {
    if (Test-Path $TempDir) {
        try {
            Remove-Item -Path $TempDir -Recurse -Force -ErrorAction Stop
            Write-Host "  Временная папка удалена: tmp\"
        }
        catch {
            Write-Warning "Не удалось удалить временную папку tmp\ (будет перезаписана)"
        }
    }
}

function New-TempFolder {
    Clear-TempFolder
    try {
        New-Item -Path $TempDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        Write-Host "  Временная папка создана: tmp\"
    }
    catch {
        Exit-WithPause -Message @"
Не удалось создать временную папку.

Путь: $TempDir
Причина: $($_.Exception.Message)
"@ -ExitCode 99
    }
}

# ============================================================
# Шаг 1. Проверка и копирование конфигурационного файла
# ============================================================
Write-Step "Шаг 1/7: Проверка конфигурационного файла"

$configFile = Join-Path $ProjectRoot "config.yaml"
$configSample = Join-Path $ProjectRoot "config.yaml.sample"

if (-not (Test-Path $configFile)) {
    Write-Host "  Файл config.yaml не найден."
    
    if (Test-Path $configSample) {
        try {
            Copy-Item -Path $configSample -Destination $configFile -Force -ErrorAction Stop
            Write-Success "Создан config.yaml из config.yaml.sample"
        }
        catch {
            Exit-WithPause -Message @"
Не удалось создать config.yaml из образца.

Образец: $configSample
Целевой файл: $configFile
Причина: $($_.Exception.Message)
"@ -ExitCode 1
        }
    }
    else {
        Exit-WithPause -Message @"
Файл config.yaml отсутствует, и образец config.yaml.sample не найден.

Необходимо наличие хотя бы одного из этих файлов для работы программы.
Свяжитесь с разработчиком.
"@ -ExitCode 1
    }
}
else {
    Write-Success "Файл config.yaml уже существует."
}

# ============================================================
# Шаг 2. Проверка наличия Visual C++ Redistributable
# ============================================================
Write-Step "Шаг 2/7: Проверка Visual C++ Redistributable"

$vcRedistInstalled = $false
try {
    # Пытаемся загрузить системную DLL, которая устанавливается пакетом VC++ Redistributable
    $dllPath = "vcruntime140.dll"
    $result = [System.Runtime.InteropServices.Marshal]::LoadLibrary($dllPath)
    if ($result -ne [IntPtr]::Zero) {
        $vcRedistInstalled = $true
        Write-Success "Visual C++ Redistributable найден."
    }
}
catch {
    # Игнорируем ошибку — проверим другим способом ниже
}

# Дополнительная проверка через поиск файла в System32
if (-not $vcRedistInstalled) {
    $system32Path = Join-Path $env:SystemRoot "System32\vcruntime140.dll"
    $sysWow64Path = Join-Path $env:SystemRoot "SysWOW64\vcruntime140.dll"
    
    if ((Test-Path $system32Path) -or (Test-Path $sysWow64Path)) {
        $vcRedistInstalled = $true
        Write-Success "Visual C++ Redistributable найден (System32)."
    }
}

if (-not $vcRedistInstalled) {
    Write-Warning "Visual C++ Redistributable не найден."
    Write-Host ""
    Write-Host "  Для работы программы требуется Microsoft Visual C++ Redistributable."
    Write-Host "  Это обязательный компонент, который нужен для модуля sherpa-onnx."
    Write-Host ""
    Write-Host "  Пожалуйста, установите его вручную, следуя инструкциям на сайте Microsoft,"
    Write-Host "  а затем запустите этот скрипт установки заново."
    Write-Host ""
    Write-Host "  Сейчас откроется сайт Microsoft для скачивания..."
    
    # Открываем ссылку в браузере по умолчанию
    try {
        Start-Process "https://aka.ms/vc14/vc_redist.x64.exe"
        Write-Host "  Ссылка открыта в браузере."
    }
    catch {
        Write-Host "  Не удалось открыть браузер автоматически."
        Write-Host "  Пожалуйста, перейдите по ссылке вручную:"
        Write-Host "  https://aka.ms/vc14/vc_redist.x64.exe"
    }
    
    Exit-WithPause -Message @"
Установка прервана: требуется Microsoft Visual C++ Redistributable.

Установите его и запустите скрипт заново.
"@ -ExitCode 9
}

# ============================================================
# Шаг 3. Установка портативного Python
# ============================================================
Write-Step "Шаг 3/7: Установка портативного Python $PythonVersion"

if (Test-Path $PythonExe) {
    Write-Success "Python уже установлен в bin\python\"
}
else {
    Write-Host "  Python не найден. Начинаю скачивание и установку..."
    
    # Создаём временную папку
    New-TempFolder
    
    # Скачиваем портативный Python
    $pythonArchive = Join-Path $TempDir $PythonZip
    Write-Host "  Скачивание Python $PythonVersion..."
    
    try {
        Invoke-WebRequest -Uri $PythonUrl -OutFile $pythonArchive -ErrorAction Stop
        $archiveSize = (Get-Item $pythonArchive).Length
        Write-Success "Python скачан (размер: $([math]::Round($archiveSize/1MB, 1)) МБ)"
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скачать портативный Python.

Ссылка: $PythonUrl
Причина: $($_.Exception.Message)

Проверьте подключение к интернету и повторите попытку.
"@ -ExitCode 2
    }
    
    # Создаём папку для Python
    try {
        if (-not (Test-Path $PythonDir)) {
            New-Item -Path $PythonDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        }
        Write-Host "  Распаковка Python в bin\python\..."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось создать папку для Python.

Путь: $PythonDir
Причина: $($_.Exception.Message)
"@ -ExitCode 2
    }
    
    # Распаковываем архив
    try {
        Expand-Archive -Path $pythonArchive -DestinationPath $PythonDir -Force -ErrorAction Stop
        Write-Success "Python распакован."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось распаковать архив Python.

Архив: $pythonArchive
Папка назначения: $PythonDir
Причина: $($_.Exception.Message)
"@ -ExitCode 2
    }
    
    # Удаляем архив
    Remove-Item -Path $pythonArchive -Force -ErrorAction SilentlyContinue
    
    # ============================================================
    # Настройка файла python314._pth
    # ============================================================
    Write-Host "  Настройка путей поиска модулей..."
    
    $pthFile = Join-Path $PythonDir "python314._pth"
    $pthContent = @"
python314.zip
.
Lib
Lib/site-packages
../../src
import site
"@
    
    try {
        # Используем Set-Content с UTF-8 без BOM для корректной работы Python
        #$pthContent | Set-Content -Path $pthFile -Encoding UTF8 -Force -ErrorAction Stop
        # Использование .NET гарантирует UTF-8 без BOM
        [System.IO.File]::WriteAllLines($pthFile, $pthContent)
        Write-Success "Файл python314._pth настроен."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось настроить файл python314._pth.

Путь: $pthFile
Причина: $($_.Exception.Message)
"@ -ExitCode 2
    }
    
    # ============================================================
    # Установка pip
    # ============================================================
    Write-Host "  Установка pip..."
    
    $getPipPath = Join-Path $PythonDir "get-pip.py"
    
    try {
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath -ErrorAction Stop
        Write-Host "  Скачан get-pip.py"
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скачать установщик pip.

Причина: $($_.Exception.Message)

Проверьте подключение к интернету.
"@ -ExitCode 3
    }
    
    # Запускаем установку pip
    $pipProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList $getPipPath `
        -NoNewWindow `
        -Wait `
        -PassThru
    
    # Удаляем get-pip.py в любом случае
    Remove-Item -Path $getPipPath -Force -ErrorAction SilentlyContinue
    
    # Проверяем код возврата
    if ($pipProcess.ExitCode -ne 0) {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось установить pip.

Код возврата: $($pipProcess.ExitCode)

Попробуйте запустить скрипт повторно.
"@ -ExitCode 3
    }
    
    Write-Success "pip установлен успешно."
    
    # Очищаем временную папку после установки Python
    Clear-TempFolder
}

# ============================================================
# Шаг 4. Установка или обновление пакетов Python
# ============================================================
Write-Step "Шаг 4/7: Установка зависимостей Python"

$requirementsFile = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $requirementsFile)) {
    Write-ErrorMsg "Файл requirements.txt не найден."
    Exit-WithPause -Message @"
Файл requirements.txt отсутствует в папке проекта.

Путь: $requirementsFile

Этот файл содержит список необходимых пакетов Python.
Без него невозможно установить зависимости для работы программы.

Убедитесь, что репозиторий склонирован полностью.
Свяжитесь с разработчиком, если ошибка повторяется.
"@ -ExitCode 4
}

Write-Host "  Установка пакетов из requirements.txt..."
Write-Host "  Это может занять несколько минут..."
Write-Host ""

# Запускаем pip install
$pipProcess = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m pip install -r `"$requirementsFile`"" `
    -NoNewWindow `
    -Wait `
    -PassThru

if ($pipProcess.ExitCode -ne 0) {
    Exit-WithPause -Message @"
Не удалось установить зависимости Python.

Код возврата pip: $($pipProcess.ExitCode)

Возможные причины:
  1. Ошибка в файле requirements.txt (неверные версии пакетов)
  2. Отсутствует подключение к интернету
  3. Требуемый пакет недоступен в PyPI

Проверьте сообщения выше и попробуйте запустить скрипт повторно.
"@ -ExitCode 4
}

Write-Success "Зависимости Python установлены успешно."

# Очищаем временную папку
Clear-TempFolder

# ============================================================
# Шаг 5. Проверка и установка ffmpeg
# ============================================================
Write-Step "Шаг 5/7: Проверка и установка ffmpeg"

if (Test-Path $FfmpegExe) {
    Write-Success "ffmpeg уже установлен в bin\ffmpeg.exe"
}
else {
    Write-Host "  ffmpeg не найден. Начинаю скачивание и установку..."
    
    # Создаём временную папку
    New-TempFolder
    
    # Скачиваем ffmpeg
    $ffmpegArchive = Join-Path $TempDir "ffmpeg.zip"
    Write-Host "  Скачивание ffmpeg..."
    
    try {
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $ffmpegArchive -ErrorAction Stop
        $archiveSize = (Get-Item $ffmpegArchive).Length
        Write-Success "ffmpeg скачан (размер: $([math]::Round($archiveSize/1MB, 1)) МБ)"
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скачать ffmpeg.

Ссылка: $FfmpegUrl
Причина: $($_.Exception.Message)

Проверьте подключение к интернету и повторите попытку.
"@ -ExitCode 5
    }
    
    # Распаковываем архив
    $ffmpegExtractPath = Join-Path $TempDir "ffmpeg_extract"
    Write-Host "  Распаковка ffmpeg..."
    
    try {
        if (Test-Path $ffmpegExtractPath) {
            Remove-Item -Path $ffmpegExtractPath -Recurse -Force
        }
        Expand-Archive -Path $ffmpegArchive -DestinationPath $ffmpegExtractPath -Force -ErrorAction Stop
        Write-Success "Архив ffmpeg распакован."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось распаковать архив ffmpeg.

Архив: $ffmpegArchive
Причина: $($_.Exception.Message)
"@ -ExitCode 5
    }
    
    # Ищем ffmpeg.exe внутри распакованной папки
    Write-Host "  Поиск ffmpeg.exe в распакованном архиве..."
    $extractedFfmpeg = Get-ChildItem -Path $ffmpegExtractPath -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if (-not $extractedFfmpeg) {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось найти ffmpeg.exe в распакованном архиве.

Путь поиска: $ffmpegExtractPath

Возможно, изменилась структура архива ffmpeg.
Свяжитесь с разработчиком.
"@ -ExitCode 5
    }
    
    Write-Host "  Найден: $($extractedFfmpeg.FullName)"
    
    # Создаём папку bin, если её ещё нет (на случай, если Python ещё не ставился)
    if (-not (Test-Path $FfmpegDir)) {
        New-Item -Path $FfmpegDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        Write-Host "  Создана папка bin\"
    }
    
    # Копируем ffmpeg.exe в bin\
    try {
        Copy-Item -Path $extractedFfmpeg.FullName -Destination $FfmpegExe -Force -ErrorAction Stop
        Write-Success "ffmpeg.exe скопирован в bin\ffmpeg.exe"
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скопировать ffmpeg.exe.

Источник: $($extractedFfmpeg.FullName)
Назначение: $FfmpegExe
Причина: $($_.Exception.Message)
"@ -ExitCode 5
    }
    
    # Очищаем временную папку
    Clear-TempFolder
    
    # Финальная проверка
    if (Test-Path $FfmpegExe) {
        Write-Success "Установка ffmpeg завершена успешно."
    }
    else {
        Exit-WithPause -Message @"
Ошибка установки ffmpeg: файл $FfmpegExe не обнаружен после копирования.

Попробуйте запустить скрипт повторно.
"@ -ExitCode 5
    }
}

# ============================================================
# Шаг 6. Загрузка ASR-модели (GigaAM v3)
# ============================================================
Write-Step "Шаг 6/7: Загрузка ASR-модели GigaAM v3"

# Проверяем наличие всех ключевых файлов модели
$asrModelComplete = (Test-Path $AsrModelFile1) -and (Test-Path $AsrModelFile2)

if ($asrModelComplete) {
    Write-Success "ASR-модель уже установлена в models\asr\giga-am-v3\"
    Write-Host "  Найдены: model.int8.onnx, tokens.txt"
}
else {
    if (Test-Path $AsrModelDir) {
        Write-Warning "Папка модели существует, но не все файлы на месте."
        Write-Host "  Будет выполнена повторная загрузка."
    }
    else {
        Write-Host "  ASR-модель не найдена. Начинаю скачивание..."
    }
    
    # Создаём временную папку
    New-TempFolder
    
    # Скачиваем модель
    $asrArchive = Join-Path $TempDir "asr_model.tar.bz2"
    Write-Host "  Скачивание ASR-модели (это может занять время, размер ~150 МБ)..."
    
    try {
        Invoke-WebRequest -Uri $AsrModelUrl -OutFile $asrArchive -ErrorAction Stop
        $archiveSize = (Get-Item $asrArchive).Length
        Write-Success "ASR-модель скачана (размер: $([math]::Round($archiveSize/1MB, 1)) МБ)"
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скачать ASR-модель.

Ссылка: $AsrModelUrl
Причина: $($_.Exception.Message)

Проверьте подключение к интернету и повторите попытку.
Модель имеет большой размер, убедитесь, что соединение стабильно.
"@ -ExitCode 6
    }
    
    # Распаковка ASR-модели

    # 1. Портативная установка 7-Zip (если еще не установлен)
    # Проверяем/устанавливаем портативный 7-Zip
    $sevenZipDir = "bin\7zip"
    $sevenZipExe = Join-Path $sevenZipDir "7z.exe"
    
    if (-not (Test-Path $sevenZipExe)) {
        Write-Host "  Портативная установка 7-Zip..."
        
        # Создаём папку для 7-Zip, если её нет
        if (-not (Test-Path $sevenZipDir)) {
            New-Item -Path $sevenZipDir -ItemType Directory -Force | Out-Null
        }
        
        $installerUrl = "https://github.com/ip7z/7zip/releases/download/26.01/7z2601-x64.exe"
        $installerPath = Join-Path $TempDir "7z-installer.exe"
        $extractTempDir = Join-Path $TempDir "7z-extract"
        
        try {
            Write-Host "  Скачивание установщика 7-Zip..."
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -ErrorAction Stop
            
            Write-Host "  Извлечение 7-Zip (тихая распаковка)..."
            Start-Process -FilePath $installerPath -ArgumentList "/S /D=$extractTempDir" -Wait -NoNewWindow
            
            # Копируем только консольную версию и библиотеку
            Copy-Item (Join-Path $extractTempDir "7z.exe") -Destination $sevenZipDir -ErrorAction Stop
            Copy-Item (Join-Path $extractTempDir "7z.dll") -Destination $sevenZipDir -ErrorAction Stop
            
            Write-Success "7-Zip портативный установлен."
        }
        catch {
            Clear-TempFolder
            Exit-WithPause -Message @"
Не удалось установить 7-Zip.

Причина: $($_.Exception.Message)

Проверьте подключение к интернету и доступность GitHub.
"@ -ExitCode 6
        }
        finally {
            # Убираем временные файлы
            if (Test-Path $installerPath) {
                Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
            }
            if (Test-Path $extractTempDir) {
                Remove-Item $extractTempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
    else {
        Write-Host "  7-Zip уже установлен в bin\7zip\"
    }

    # 2. Распаковка модели скачанным 7-Zip
    $asrExtractPath = Join-Path $TempDir "asr_extract"
    try {
        if (Test-Path $asrExtractPath) {
            Remove-Item -Path $asrExtractPath -Recurse -Force
        }
        New-Item -Path $asrExtractPath -ItemType Directory -Force | Out-Null

        # Шаг 2.1: Распаковка .tar.bz2 → .tar
        Write-Host "  Распаковка ASR-модели (это может занять время)..."
        $sevenZipArgs = "x `"$asrArchive`" -o`"$asrExtractPath`" -y"
        $output = & $sevenZipExe $sevenZipArgs.Split() 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            throw "Ошибка распаковки .tar.bz2 (код: $LASTEXITCODE). Вывод: $($output -join ' ')"
        }
        Write-Host "  Архив .tar.bz2 распакован."

        # Шаг 2.2: Поиск и распаковка вложенного .tar
        $tarFile = Get-ChildItem -Path $asrExtractPath -Filter "*.tar" -File | Select-Object -First 1
        if (-not $tarFile) {
            throw "В распакованном архиве не найден файл .tar"
        }
        
        Write-Host "  Обнаружен вложенный архив: $($tarFile.Name)"
        $tarExtractPath = Join-Path $TempDir "tar_extract"
        if (Test-Path $tarExtractPath) {
            Remove-Item -Path $tarExtractPath -Recurse -Force
        }
        New-Item -Path $tarExtractPath -ItemType Directory -Force | Out-Null

        $tarArgs = "x `"$($tarFile.FullName)`" -o`"$tarExtractPath`" -y"
        $output = & $sevenZipExe $tarArgs.Split() 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            throw "Ошибка распаковки .tar (код: $LASTEXITCODE). Вывод: $($output -join ' ')"
        }
        
        # Переопределяем путь для копирования: теперь это содержимое распакованного tar
        $asrExtractPath = $tarExtractPath
        Write-Success "ASR-модель полностью распакована."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось распаковать ASR-модель.

Причина: $($_.Exception.Message)

Возможные причины:
1. Архив повреждён (попробуйте запустить скрипт повторно)
2. Недостаточно места на диске (нужно ~1.5 ГБ свободного места)
3. Ошибка в структуре архива
"@ -ExitCode 6
    }
    
    # Создаём папку для модели
    if (-not (Test-Path $AsrModelDir)) {
        try {
            New-Item -Path $AsrModelDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
            Write-Host "  Создана папка для модели: models\asr\giga-am-v3\"
        }
        catch {
            Clear-TempFolder
            Exit-WithPause -Message @"
Не удалось создать папку для ASR-модели.

Путь: $AsrModelDir
Причина: $($_.Exception.Message)
"@ -ExitCode 6
        }
    }
    
    # 3. Копируем все файлы, кроме папки test_wavs
    Write-Host "  Копирование файлов модели..."
    
    try {
        # Проверяем, есть ли внутри архива вложенная папка
        $extractedDirs = @(Get-ChildItem -Path $asrExtractPath -Directory)
        $extractedFiles = @(Get-ChildItem -Path $asrExtractPath -File)
        
        # Определяем, откуда копировать
        if ($extractedDirs.Count -eq 1 -and $extractedFiles.Count -eq 0) {
            # Архив содержит одну папку — берём её содержимое
            $sourcePath = $extractedDirs[0].FullName
            Write-Host "  Обнаружена вложенная папка: $($extractedDirs[0].Name)"
        }
        elseif ($extractedDirs.Count -ge 1 -or $extractedFiles.Count -ge 1) {
            # Файлы лежат прямо в корне распаковки
            $sourcePath = $asrExtractPath
            Write-Host "  Файлы модели в корне архива."
        }
        else {
            throw "Архив пуст или имеет неожиданную структуру."
        }
        
        # Копируем всё, кроме test_wavs
        Get-ChildItem -Path $sourcePath -Exclude "test_wavs" | ForEach-Object {
            $destination = Join-Path $AsrModelDir $_.Name
            Copy-Item -Path $_.FullName -Destination $destination -Recurse -Force -ErrorAction Stop
            Write-Host "    Скопирован: $($_.Name)"
        }
        
        Write-Success "Файлы ASR-модели скопированы."
    }
    catch {
        Clear-TempFolder
        Exit-WithPause -Message @"
Не удалось скопировать файлы ASR-модели.

Причина: $($_.Exception.Message)
"@ -ExitCode 6
    }
    
    # 4. Удаляем временные файлы
    Remove-Item -Path $asrArchive -Force -ErrorAction SilentlyContinue
    if (Test-Path $asrExtractPath) {
        Remove-Item -Path $asrExtractPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # 5. Очищаем временную папку
    Clear-TempFolder
    
    # 6. Финальная проверка
    $asrModelComplete = (Test-Path $AsrModelFile1) -and (Test-Path $AsrModelFile2)
    if ($asrModelComplete) {
        Write-Success "Установка ASR-модели завершена успешно."
    }
    else {
        Exit-WithPause -Message @"
Ошибка установки ASR-модели: не все файлы обнаружены после копирования.

Ожидаемые файлы:
  - $AsrModelFile1
  - $AsrModelFile2

Попробуйте запустить скрипт повторно.
"@ -ExitCode 6
    }
}

# ============================================================
# Шаг 7. Загрузка VAD-модели (Silero)
# ============================================================
Write-Step "Шаг 7/7: Загрузка VAD-модели Silero"

if (Test-Path $VadModelFile) {
    Write-Success "VAD-модель уже установлена в models\vad\silero\"
}
else {
    Write-Host "  VAD-модель не найдена. Начинаю скачивание..."
    
    # Создаём папку для VAD-модели
    $vadModelDir = Split-Path $VadModelFile -Parent
    if (-not (Test-Path $vadModelDir)) {
        try {
            New-Item -Path $vadModelDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
            Write-Host "  Создана папка для VAD-модели: models\vad\silero\"
        }
        catch {
            Exit-WithPause -Message @"
Не удалось создать папку для VAD-модели.

Путь: $vadModelDir
Причина: $($_.Exception.Message)
"@ -ExitCode 7
        }
    }
    
    # Скачиваем модель Silero VAD напрямую в папку назначения
    Write-Host "  Скачивание VAD-модели Silero..."
    
    try {
        Invoke-WebRequest -Uri $VadModelUrl -OutFile $VadModelFile -ErrorAction Stop
        $fileSize = (Get-Item $VadModelFile).Length
        Write-Success "VAD-модель скачана (размер: $([math]::Round($fileSize/1KB, 1)) КБ)"
    }
    catch {
        # Удаляем возможно недокачанный файл
        if (Test-Path $VadModelFile) {
            Remove-Item -Path $VadModelFile -Force -ErrorAction SilentlyContinue
        }
        
        Exit-WithPause -Message @"
Не удалось скачать VAD-модель.

Ссылка: $VadModelUrl
Причина: $($_.Exception.Message)

Проверьте подключение к интернету и повторите попытку.
"@ -ExitCode 7
    }
    
    # Финальная проверка
    if (Test-Path $VadModelFile) {
        Write-Success "Установка VAD-модели завершена успешно."
    }
    else {
        Exit-WithPause -Message @"
Ошибка установки VAD-модели: файл $VadModelFile не обнаружен после скачивания.

Попробуйте запустить скрипт повторно.
"@ -ExitCode 7
    }
}

# ============================================================
# Шаг 8. Завершение
# ============================================================

# Финальная очистка временной папки
Clear-TempFolder

Write-Host ""
Write-Host ("=" * 60)
Write-Host ""
Write-Host "  Установка успешно завершена!" -ForegroundColor Green
Write-Host ""
Write-Host "  Все компоненты установлены в портабельном режиме"
Write-Host "  в папку проекта и не требуют прав администратора."
Write-Host ""
Write-Host "  Установлено:"
Write-Host "    - config.yaml (конфигурация)"
Write-Host "    - bin\python\ (Python $PythonVersion с pip)"
Write-Host "    - bin\ffmpeg.exe (ffmpeg)"
Write-Host "    - models\asr\giga-am-v3\ (ASR-модель GigaAM v3)"
Write-Host "    - models\vad\silero\ (VAD-модель Silero)"
Write-Host ""
Write-Host "  Теперь можно запустить распознавание, перетащив"
Write-Host "  аудиофайл на ярлык ""Распознать аудио""."
Write-Host ""

Exit-WithPause -Message "Инициализация окружения успешно завершена." -ExitCode 0
