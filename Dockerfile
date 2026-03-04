# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем ffmpeg для обработки видео
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем uv для управления зависимостями
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости
RUN uv sync --frozen --no-dev

# Копируем остальные файлы приложения
COPY app.py ./
COPY prompts ./prompts
COPY static ./static

# Создаем директории для временных файлов
RUN mkdir -p uploads temp

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uv", "run", "python", "app.py"]