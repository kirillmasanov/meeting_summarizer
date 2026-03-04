# 🎥 Meeting Summarizer

Веб-приложение для автоматической обработки видеозаписей встреч с извлечением аудио, асинхронной транскрибацией через Yandex SpeechKit API v3 и суммаризацией через AI Studio Model Gallery на основе настраиваемого системного промпта.

## 🌟 Возможности

- **Загрузка видео**: Drag-and-drop интерфейс для загрузки видеофайлов (MP4, WebM, до 1GB и длительностью до 4 часов)
- **Извлечение аудио**: Автоматическое извлечение аудиодорожки из видео с помощью ffmpeg
- **Асинхронная транскрибация**: Преобразование речи в текст через Yandex SpeechKit API v3
- **Суммаризация**: Создание структурированного резюме через AI Studio Model Gallery
- **Типовые промпты**: Выбор типа резюме (встреча с клиентом, вебинар, собеседование, стендап, продажная встреча) и при необходимости правка текста промпта
- **Копирование результатов**: Быстрое копирование резюме в буфер обмена
- **Health check**: Эндпоинт `/api/health` для проверки состояния сервиса

## 📋 Требования

### Системные зависимости

- **Python 3.8+**
- **ffmpeg** - для извлечения аудио из видео
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`

### Yandex Cloud

- Аккаунт в [Yandex Cloud](https://cloud.yandex.ru/)
- API ключ сервисного аккаунта для доступа к Yandex SpeechKit
- ID каталога (Folder ID)

## 🚀 Установка

**Установка uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Добавьте uv в PATH:
Для Bash:
```bash
echo 'export PATH="$HOME/.uv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
Для Zsh:
```bash
echo 'export PATH="$HOME/.uv/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Установка проекта:**
```bash
git clone https://github.com/kirillmasanov/meeting_summarizer
cd meeting_summarizer

# Создание виртуального окружения и установка зависимостей
uv sync
```

**Примечание:** При использовании `uv` не требуется вручную активировать виртуальное окружение. Команда `uv run` автоматически использует созданное окружение.

### Настройка Yandex Cloud

#### 1. Получение Folder ID

Folder ID можно получить с помощью yc cli:
```bash
yc config get folder-id
```

Или в веб-консоли:
1. Перейдите в [консоль Yandex Cloud](https://console.cloud.yandex.ru/)
2. Выберите ваш каталог
3. ID каталога отображается в URL или в информации о каталоге

#### 2. Создание сервисного аккаунта

**Через веб-консоль:**
1. Перейдите в раздел "Сервисные аккаунты"
2. Нажмите "Создать сервисный аккаунт"
3. Укажите имя (например, `meeting-summarizer-sa`)
4. Добавьте описание (опционально)

**Через CLI:**
```bash
# Создание сервисного аккаунта
yc iam service-account create \
  --name meeting-summarizer-sa \
  --description "Service account for meeting summarizer app"

# Получение ID созданного аккаунта
SA_ID=$(yc iam service-account get meeting-summarizer-sa --format json | jq -r '.id')
echo "Service Account ID: $SA_ID"
```

#### 3. Назначение ролей

Сервисному аккаунту необходимы следующие роли:
- `ai.speechkit-stt.user` - для работы с SpeechKit API (асинхронная транскрибация)
- `ai.languageModels.user` - для работы с AI Studio Model Gallery (суммаризация)

**Через веб-консоль:**
1. Откройте страницу каталога
2. Перейдите на вкладку "Права доступа"
3. Нажмите "Назначить роли"
4. Выберите сервисный аккаунт
5. Добавьте обе роли: `ai.speechkit-stt.user` и `ai.languageModels.user`

**Через CLI:**
```bash
# Получите FOLDER_ID
FOLDER_ID=$(yc config get folder-id)

# Назначьте роли
yc resource-manager folder add-access-binding $FOLDER_ID \
  --role ai.speechkit-stt.user \
  --service-account-id $SA_ID

yc resource-manager folder add-access-binding $FOLDER_ID \
  --role ai.languageModels.user \
  --service-account-id $SA_ID
```

#### 4. Создание API ключа

**Через веб-консоль:**
1. Откройте страницу сервисного аккаунта
2. Перейдите на вкладку "API-ключи"
3. Нажмите "Создать API-ключ"
4. Скопируйте ключ (он показывается только один раз!)

**Через CLI:**
```bash
# Создание API-ключа
yc iam api-key create --service-account-id $SA_ID --format json

# Сохранение ключа в переменную
API_KEY=$(yc iam api-key create --service-account-id $SA_ID --format json | jq -r '.secret')
```

#### 5. Настройка переменных окружения

##### Способ 1: Через переменные окружения

Создайте файл `.env` в корне проекта:

```env
YANDEX_API_KEY=your_api_key_here
YANDEX_FOLDER_ID=your_folder_id_here
YANDEX_CLOUD_MODEL=qwen3-235b-a22b-fp8/latest
SERVER_PORT=8000
```

**Примечание:** Параметры `YANDEX_CLOUD_MODEL` и `SERVER_PORT` опциональны. Без них используются значения по умолчанию. `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` обязательны для работы приложения.

**Автоматическое создание .env через CLI:**
```bash
# Получение значений
FOLDER_ID=$(yc config get folder-id)
SA_ID=$(yc iam service-account get meeting-summarizer-sa --format json | jq -r '.id')
API_KEY=$(yc iam api-key create --service-account-id $SA_ID --format json | jq -r '.secret')

# Создание .env файла
cat > .env << EOF
YANDEX_API_KEY=$API_KEY
YANDEX_FOLDER_ID=$FOLDER_ID
YANDEX_CLOUD_MODEL=qwen3-235b-a22b-fp8/latest
SERVER_PORT=8000
EOF
```

### Проверка установки ffmpeg

```bash
ffmpeg -version
```

Если команда выполнена успешно, ffmpeg установлен корректно.

## 🎯 Запуск приложения

### Вариант 1: Локальный запуск

```bash
uv run python app.py
```

Приложение будет доступно по адресу: **http://localhost:8000**

#### Режим разработки с автоперезагрузкой:
```bash
uv run uvicorn app:app --reload
```

### Вариант 2: Запуск в Docker

**Требования:**
- Docker
- Docker Compose

**Запуск:**

```bash
# Сборка и запуск контейнера
docker compose up -d

# Просмотр логов
docker compose logs -f

# Остановка контейнера
docker compose down
```

Приложение будет доступно по адресу: **http://localhost:8000**

**Примечание:** При использовании Docker все зависимости (включая ffmpeg) устанавливаются автоматически внутри контейнера.

### Проверка работоспособности

Откройте в браузере: **http://localhost:8000**

Вы должны увидеть веб-интерфейс с формой загрузки файлов.

## 📖 Использование

### Веб-интерфейс

#### Обработка видео

1. Убедитесь, что в `.env` заданы `YANDEX_API_KEY` и `YANDEX_FOLDER_ID`, и откройте **http://localhost:8000**
2. Выберите тип промпта кнопкой (Встреча с клиентом, Обучающий вебинар, Собеседование, Стендап, Продажная встреча). При необходимости отредактируйте текст в блоке «Настройки промпта»
3. Перетащите видеофайл в область загрузки или нажмите для выбора
4. Нажмите «Загрузить и обработать»
5. Дождитесь завершения (спиннер и стадии: загрузка → извлечение аудио → транскрибация и суммаризация)
6. Просмотрите резюме, при необходимости скопируйте в буфер обмена
7. Нажмите «Обработать новый файл» для следующей задачи

**Проверка состояния сервиса:** `GET /api/health` — возвращает статус, наличие ffmpeg и настройку API ключей.

## 🏗️ Структура проекта

```
meeting_summarizer/
├── app.py                      # Основное FastAPI приложение
├── pyproject.toml              # Конфигурация проекта и зависимости
├── uv.lock                     # Lockfile зависимостей
├── .env                        # Переменные окружения (не в git)
├── .gitignore                  # Игнорируемые файлы для Git
├── Dockerfile                  # Конфигурация Docker образа
├── docker-compose.yml          # Конфигурация Docker Compose
├── .dockerignore               # Игнорируемые файлы для Docker
├── README.md                   # Документация
├── prompts/                    # Типовые промпты для суммаризации
│   ├── meeting.txt             # Встреча с клиентом
│   ├── webinar.txt             # Обучающий вебинар
│   ├── interview.txt           # Собеседование
│   ├── standup.txt             # Стендап
│   └── sales.txt               # Продажная встреча
├── static/                     # Статические файлы
│   └── index.html             # Веб-интерфейс
├── uploads/                    # Загруженные видео (создается автоматически)
└── temp/                       # Временные файлы (создается автоматически)
```

## ⚙️ Конфигурация

### Ограничения

- **Максимальный размер файла**: 1 GB
- **Максимальная длительность видео**: 4 часа (240 минут)
- **Поддерживаемые форматы**: MP4, WebM (Zoom, Телемост)
- **Хранение данных**: Файлы и статусы удаляются сразу после получения результата
- **Аудио формат**: Автоматическое преобразование в MP3

### Настройка переменных окружения

Приложение поддерживает следующие переменные окружения в файле `.env`:

- `YANDEX_API_KEY` (обязательно) — API ключ сервисного аккаунта Yandex Cloud
- `YANDEX_FOLDER_ID` (обязательно) — ID каталога в Yandex Cloud
- `YANDEX_CLOUD_MODEL` (опционально, по умолчанию: `qwen3-235b-a22b-fp8/latest`) — модель для суммаризации
- `SERVER_PORT` (опционально, по умолчанию: `8000`) — порт сервера

### Настройка системного промпта

Типовые промпты загружаются из папки `prompts/` при запуске. В интерфейсе доступны кнопки выбора типа:

- **Встреча с клиентом** (`prompts/meeting.txt`)
- **Обучающий вебинар** (`prompts/webinar.txt`)
- **Собеседование** (`prompts/interview.txt`)
- **Стендап** (`prompts/standup.txt`)
- **Продажная встреча** (`prompts/sales.txt`)

При нажатии «Загрузить и обработать» используется текущий текст из блока «Настройки промпта» (выбранный тип или отредактированный вручную). Чтобы изменить текст типового промпта навсегда, отредактируйте соответствующий файл в `prompts/` и перезапустите приложение. 

## 🐳 Docker

### Особенности Docker-версии

- **Автоматическая установка ffmpeg** - не требуется устанавливать вручную
- **Изолированное окружение** - все зависимости внутри контейнера
- **Простое развертывание** - один файл `.env` (опционально) и команда `docker compose up`
- **Персистентность данных** - директории `uploads/` и `temp/` монтируются как volumes

### Управление контейнером

```bash
# Пересборка образа после изменений
docker compose build

# Запуск в фоновом режиме
docker compose up -d

# Просмотр логов
docker compose logs -f meeting-summarizer

# Остановка без удаления контейнера
docker compose stop

# Остановка и удаление контейнера
docker compose down

# Полная очистка (включая volumes)
docker compose down -v
```

### Настройка портов

По умолчанию приложение доступно на порту 8000. Чтобы изменить порт, укажите `SERVER_PORT` в файле `.env`:

```env
SERVER_PORT=3000
```

Затем пересоздайте контейнер:
```bash
docker compose up -d --force-recreate
```

## 🔄 Обновление приложения на виртуальной машине

Подключитесь к вашей ВМ по SSH и выполните следующие команды:

```bash
# 1. Перейдите в директорию проекта
cd /path/to/meeting-summarizer

# 2. Остановите текущий контейнер
docker compose down

# 3. Получите последние изменения из GitHub
git pull origin main

# 4. Пересоберите и запустите контейнер с новой версией
docker compose up -d --build

# 5. Проверьте логи, чтобы убедиться что все работает
docker compose logs -f meeting-summarizer
```

### Проверка обновления

После обновления убедитесь, что все работает корректно:

```bash
# Проверьте последний коммит
git log -1 --oneline

# Проверьте статус контейнера
docker compose ps

# Откройте приложение в браузере
# http://<IP-вашей-ВМ>:8000
```

### Важные моменты

- **Файл .env сохраняется**: Настройки в `.env` не затрагиваются при обновлении
- **Данные сохраняются**: Директории `uploads/` и `temp/` (если есть) сохраняются благодаря volumes

### Откат к предыдущей версии

Если что-то пошло не так, можно откатиться к предыдущей версии:

```bash
# Посмотрите список коммитов
git log --oneline

# Откатитесь к нужному коммиту
git checkout <commit-hash>

# Пересоберите контейнер
docker compose up -d --build
```

## 🔒 Безопасность

### Рекомендации для production

1. **Используйте HTTPS**: Настройте reverse proxy (nginx/traefik) с SSL сертификатом
2. **Ограничьте доступ**: Используйте firewall или VPN для ограничения доступа
3. **Храните ключи в .env**: Используйте переменные `YANDEX_API_KEY` и `YANDEX_FOLDER_ID`
4. **Мониторинг**: Отслеживайте использование API и логи приложения
5. **Регулярные обновления**: Обновляйте зависимости для устранения уязвимостей

### Хранение данных

- **API ключи**: Задаются только в `.env`, в памяти не хранятся
- **Видео файлы**: Удаляются сразу после извлечения аудио
- **Аудио файлы**: Удаляются сразу после транскрибации
- **Результаты**: Удаляются после выдачи клиенту; задачи удаляются после получения результата

## 🔗 Полезные ссылки

- [Yandex SpeechKit — обзор](https://aistudio.yandex.ru/docs/ru/speechkit/overview.html)
- [Правила тарификации SpeechKit](https://aistudio.yandex.ru/docs/ru/speechkit/pricing.html)
- [Yandex AI Studio Model Gallery](https://yandex.cloud/ru/docs/ai-studio/concepts/generation/models)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
