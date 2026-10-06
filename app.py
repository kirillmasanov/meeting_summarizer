"""
FastAPI приложение для обработки видеозаписей встреч
с извлечением аудио, транскрибацией и суммаризацией
"""

import os
import uuid
import base64
import time
import json
import asyncio
import re
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
from dotenv import load_dotenv
import subprocess
import urllib3

# Отключаем предупреждения о незащищенных HTTPS запросах
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    force=True  # Переопределяет настройки uvicorn
)
logger = logging.getLogger(__name__)

# Устанавливаем уровень для uvicorn логгера
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# Загружаем переменные окружения
load_dotenv()

# Константы
DEFAULT_API_KEY = os.getenv("YANDEX_API_KEY")
DEFAULT_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
SUMMARY_MODELS = (
    "deepseek-v4.1-flash",
    "qwen3.6-35b-a3b",
    "aliceai-llm-flash",
    "yandexgpt-5.1",
)
LLM_MODEL = os.getenv("YANDEX_CLOUD_MODEL", "deepseek-v4.1-flash")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
MAX_VIDEO_DURATION = 14400  # 4 часа в секундах
UPLOAD_TIMEOUT = 600  # 10 минут
STATUS_CHECK_TIMEOUT = 30  # 30 секунд
RESULT_TIMEOUT = 60  # 1 минута
POLL_INTERVAL = 5  # 5 секунд между проверками статуса

# Логируем наличие дефолтных ключей
if DEFAULT_API_KEY and DEFAULT_FOLDER_ID:
    logger.info("Найдены дефолтные API ключи в переменных окружения")
else:
    logger.info("Дефолтные API ключи не найдены, потребуется ввод через UI")

# Предустановленные промпты: id -> (название, путь к файлу)
PROMPTS_DIR = Path("prompts")
PROMPT_FILES = {
    "meeting": ("Встреча с клиентом", PROMPTS_DIR / "meeting.txt"),
    "webinar": ("Обучающий вебинар", PROMPTS_DIR / "webinar.txt"),
    "interview": ("Собеседование", PROMPTS_DIR / "interview.txt"),
    "standup": ("Стендап", PROMPTS_DIR / "standup.txt"),
    "sales": ("Встреча по продажам", PROMPTS_DIR / "sales.txt"),
}


def _load_prompts() -> dict:
    """Загружает все предустановленные промпты из файлов."""
    result = {}
    for prompt_id, (name, path) in PROMPT_FILES.items():
        if not path.exists():
            logger.warning(f"Файл промпта не найден: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            result[prompt_id] = {"name": name, "text": f.read().strip()}
    if not result:
        raise FileNotFoundError("Не найден ни один файл системного промпта.")
    return result


PRESET_PROMPTS = _load_prompts()
SYSTEM_PROMPT = PRESET_PROMPTS["meeting"]["text"]  # промпт по умолчанию

# Lifespan context manager для startup/shutdown событий
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not check_ffmpeg():
        logger.warning("ffmpeg не найден в системе!")
        logger.warning("Установите ffmpeg для работы приложения")
    yield
    # Shutdown (если нужно что-то делать при остановке)

# Создаем FastAPI приложение
app = FastAPI(title="Meeting Summarizer", version="1.0.0", lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директории для хранения файлов
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")

for directory in [UPLOAD_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)


# Модели данных для задач
@dataclass
class TaskStatus:
    """Статус задачи обработки"""
    task_id: str
    status: str  # pending, processing, completed, error
    stage: str  # upload, extract_audio, transcribe, completed
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filename: Optional[str] = None
    model: Optional[str] = None


# Глобальное хранилище задач
tasks: Dict[str, TaskStatus] = {}


def check_ffmpeg() -> bool:
    """Проверяет наличие ffmpeg в системе"""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_duration(video_path: Path) -> float:
    """Получает длительность видео в секундах"""
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        duration = float(result.stdout.decode('utf-8').strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Ошибка получения длительности видео: {e}")
        raise RuntimeError("Не удалось определить длительность видео")


def extract_audio_from_video(
    input_file: Path,
    output_file: Path,
    audio_format: str = 'mp3',
    audio_bitrate: str = '192k'
) -> bool:
    """Извлекает аудио из видео файла"""
    logger.info(f"Извлечение аудио из {input_file.name} ({input_file.stat().st_size / (1024*1024):.2f} MB)")
    
    # Настройка кодека в зависимости от формата
    if audio_format == 'opus':
        codec = 'libopus'
    elif audio_format == 'mp3':
        codec = 'libmp3lame'
    else:
        codec = audio_format
    
    command = [
        'ffmpeg',
        '-i', str(input_file),
        '-vn',
        '-acodec', codec,
        '-ab', audio_bitrate,
        '-ac', '1',  # моно
        '-ar', '16000',  # частота дискретизации 16kHz (оптимально для речи)
        '-y',
        str(output_file)
    ]
    
    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        audio_size = output_file.stat().st_size / (1024*1024)
        logger.info(f"Аудио извлечено: {output_file.name} ({audio_size:.2f} MB)")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore')
        logger.error(f"Ошибка ffmpeg: {error_msg}")
        raise RuntimeError(f"Ошибка извлечения аудио: {error_msg}")


def update_task_timestamp(task_id: str):
    """Обновляет временную метку задачи"""
    if task_id in tasks:
        tasks[task_id].updated_at = datetime.now()


def validate_summary_model(model: str):
    """Проверяем модель до сохранения файла и отправки аудио в облако."""
    if model.split("/")[0] == "qwen3-235b-a22b-fp8":
        raise RuntimeError(
            "Поддержка модели qwen3-235b-a22b-fp8 завершилась 30 сентября 2026 года. "
            "Выберите модель в интерфейсе или установите YANDEX_CLOUD_MODEL=deepseek-v4.1-flash "
            "в .env и перезапустите приложение."
        )
    if model not in SUMMARY_MODELS:
        raise RuntimeError("Неподдерживаемая модель. Доступны: " + ", ".join(SUMMARY_MODELS))


def transcribe_audio(
    audio_file_path: Path, system_prompt: str, task_id: str = None, model: Optional[str] = None
) -> dict:
    """Транскрибирует аудио файл через Yandex SpeechKit API v3"""
    selected_model = LLM_MODEL if model is None else model
    validate_summary_model(selected_model)
    # Используем MP3 для всех файлов
    audio_type = "MP3"
    logger.info(f"Тип аудио для распознавания: {audio_type}")
    
    # Конфигурация распознавания
    request_data = {
        "content": "",
        "recognitionModel": {
            "model": "general",
            "audioFormat": {
                "containerAudio": {
                    "containerAudioType": audio_type
                }
            },
            "languageRestriction": {
                "restrictionType": "WHITELIST",
                "languageCode": ["ru-RU"]
            },
            "textNormalization": {
                "textNormalization": "TEXT_NORMALIZATION_ENABLED",
                "phoneFormattingMode": "PHONE_FORMATTING_MODE_DISABLED",
                "profanityFilter": True,
                "literatureText": True
            }
        },
        "speakerLabeling": {
            "speakerLabeling": "SPEAKER_LABELING_DISABLED"
        },
        "summarization": {
            "modelUri": f"gpt://{DEFAULT_FOLDER_ID}/{selected_model}",
            "properties": [
                {
                    "instruction": system_prompt
                }
            ]
        }
    }
    
    # Читаем файл и кодируем в base64
    with open(audio_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
        audio_size_mb = len(audio_content) / (1024 * 1024)
        logger.info(f"Размер аудио файла для отправки: {audio_size_mb:.2f} MB")
        
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        request_data["content"] = audio_base64
    
    headers = {
        "Authorization": f"Api-key {DEFAULT_API_KEY}"
    }
    
    # Отправляем запрос
    logger.info(f"Отправка запроса в Yandex SpeechKit (размер данных: {len(audio_base64) / (1024*1024):.2f} MB)...")
    try:
        response = requests.post(
            "https://stt.api.cloud.yandex.net/stt/v3/recognizeFileAsync",
            headers=headers,
            json=request_data,
            verify=False,
            timeout=UPLOAD_TIMEOUT
        )
        logger.info(f"Получен ответ: {response.status_code}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Превышено время ожидания при отправке файла ({audio_size_mb:.2f} MB). Попробуйте файл меньшего размера.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка соединения с Yandex SpeechKit: {str(e)}")
    
    if response.status_code == 504:
        raise RuntimeError(f"Сервер Yandex SpeechKit не успел обработать запрос (504). Размер аудио: {audio_size_mb:.2f} MB. Попробуйте повторить позже или используйте файл меньшей длительности.")
    elif response.status_code == 413:
        raise RuntimeError(f"Файл слишком большой ({audio_size_mb:.2f} MB) для Yandex SpeechKit.")
    elif response.status_code != 200:
        error_text = response.text if response.text else 'Нет дополнительной информации'
        raise RuntimeError(f"Ошибка {response.status_code} от Yandex SpeechKit: {error_text}")
    
    operation_data = response.json()
    operation_id = operation_data.get("id")
    if not operation_id:
        raise RuntimeError("Operation ID not found in response")
    logger.info("SpeechKit: task_id=%s, operation_id=%s, model=%s", task_id, operation_id, selected_model)
    
    # Ожидаем завершения операции
    operation_url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
    
    while True:
        try:
            op_response = requests.get(operation_url, headers=headers, verify=False, timeout=STATUS_CHECK_TIMEOUT)
            
            if op_response.status_code == 200:
                op_data = op_response.json()
                if op_data.get("done"):
                    if "error" in op_data:
                        raise RuntimeError(f"Operation failed: {json.dumps(op_data['error'])}")
                    break
        except requests.exceptions.Timeout:
            logger.warning("Timeout при проверке статуса операции, повторная попытка...")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при проверке статуса: {e}")
        
        if task_id:
            update_task_timestamp(task_id)
        time.sleep(POLL_INTERVAL)
    
    # Получаем результаты
    try:
        speech_response = requests.get(
            f"https://stt.api.cloud.yandex.net/stt/v3/getRecognition?operation_id={operation_id}",
            headers=headers,
            verify=False,
            timeout=RESULT_TIMEOUT
        )
    except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
        raise RuntimeError(f"Ошибка при получении результатов: {str(e)}")
    
    if speech_response.status_code != 200:
        raise RuntimeError(f"Ошибка получения результатов: {speech_response.status_code}. {speech_response.text if speech_response.text else ''}")
    
    return parse_recognition_result(speech_response.text, operation_id)


def parse_recognition_result(response_text: str, operation_id: str) -> dict:
    """Проверяет поток NDJSON: HTTP 200 сам по себе не означает наличие резюме."""
    transcription_parts = []
    summary = None
    event_counts = {}
    status_codes = []

    for line_num, line in enumerate(response_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Некорректный JSON в ответе SpeechKit, строка {line_num}, operation_id={operation_id}."
            ) from e
        if not isinstance(data, dict):
            raise RuntimeError(f"Некорректный формат ответа SpeechKit, operation_id={operation_id}.")
        if "error" in data:
            error = data["error"]
            code = error.get("code") if isinstance(error, dict) else None
            code = str(code) if str(code).isdigit() else "unknown"
            # Не выводим тело ответа: оно может содержать текст встречи или секреты.
            raise RuntimeError(f"Ошибка потока SpeechKit: code={code}, operation_id={operation_id}.")

        result = data.get("result", data)
        if not isinstance(result, dict):
            raise RuntimeError(f"Некорректный формат результата SpeechKit, operation_id={operation_id}.")
        for event in ("final", "finalRefinement", "summarization", "statusCode"):
            if event in result:
                event_counts[event] = event_counts.get(event, 0) + 1
        if "statusCode" in result:
            code = result["statusCode"].get("codeType")
            code = code if code in ("WORKING", "WARNING", "CLOSED") else "UNKNOWN"
            if code not in status_codes:
                status_codes.append(code)

        if "finalRefinement" in result:
            normalized = result["finalRefinement"].get("normalizedText", {})
            alternatives = normalized.get("alternatives", [])
            if alternatives:
                text = alternatives[0].get("text", "")
                if text:
                    transcription_parts.append(text)

        if "summarization" in result:
            results_list = result["summarization"].get("results", [])
            if results_list:
                raw_summary = results_list[0].get("response", "")
                if isinstance(raw_summary, str) and raw_summary.strip():
                    summary = raw_summary.strip()

    full_transcription = " ".join(transcription_parts)
    logger.info(
        "SpeechKit result: operation_id=%s, events=%s, status_codes=%s, transcription_chars=%d, summary_chars=%d",
        operation_id, event_counts, status_codes, len(full_transcription), len(summary or ""),
    )

    if not summary:
        raise RuntimeError(
            "SpeechKit не вернул резюме. Проверьте выбранную модель, роль ai.languageModels.user "
            "и квоты модели. "
            f"operation_id={operation_id}; событий распознавания: "
            f"{event_counts.get('final', 0) + event_counts.get('finalRefinement', 0)}."
        )

    summary_text = json_to_html(summary)
    if not isinstance(summary_text, str) or not summary_text.strip():
        raise RuntimeError(f"SpeechKit вернул пустой текст резюме, operation_id={operation_id}.")
    return {
        "transcription": full_transcription,
        "summary": summary_text.strip()
    }


def json_to_html(text: str) -> str:
    """Конвертирует JSON в простой текст с форматированием"""
    if not text:
        return text
    
    # Извлекаем JSON из markdown блока
    json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    json_str = json_block_match.group(1).strip() if json_block_match else text.strip()
    
    try:
        data = json.loads(json_str)
        
        # Если это объект с ключом "text", извлекаем его значение
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        
        def format_value(value, indent=0):
            """Рекурсивное форматирование значений"""
            prefix = '  ' * indent
            
            if isinstance(value, dict):
                lines = []
                for k, v in value.items():
                    if isinstance(v, (list, dict)):
                        lines.append(f"{prefix}{k}:")
                        lines.append(format_value(v, indent + 1))
                    else:
                        lines.append(f"{prefix}{k}: {v}")
                return '\n'.join(lines)
            
            elif isinstance(value, list):
                if not value:
                    return f"{prefix}отсутствует"
                lines = []
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            lines.append(f"{prefix}• {k}: {v}")
                    else:
                        lines.append(f"{prefix}• {item}")
                return '\n'.join(lines)
            
            return f"{prefix}{value}"
        
        return format_value(data)
            
    except (json.JSONDecodeError, KeyError, TypeError):
        # Обычный текст — допустимый ответ модели, а не ошибка JSON.
        return text


async def process_video_task(
    task_id: str, video_path: Path, system_prompt: str, model: Optional[str] = None
):
    """Фоновая задача обработки видео"""
    audio_path = None
    try:
        if task_id not in tasks:
            logger.error(f"Задача {task_id} не найдена при обработке")
            return
        
        # Обновляем статус: извлечение аудио
        task = tasks[task_id]
        task.status = "processing"
        task.stage = "extract_audio"
        task.updated_at = datetime.now()
        
        # Извлекаем аудио в MP3 для всех форматов видео
        audio_path = TEMP_DIR / f"{task_id}.mp3"
        await asyncio.to_thread(extract_audio_from_video, video_path, audio_path, 'mp3', '128k')
        
        # Удаляем видеофайл сразу после извлечения аудио
        video_path.unlink(missing_ok=True)
        logger.info(f"Видеофайл удален для задачи {task_id}")
        
        # Обновляем статус: транскрибация
        task.stage = "transcribe"
        task.updated_at = datetime.now()
        
        # Транскрибируем (запускаем в отдельном потоке, чтобы не блокировать event loop)
        result = await asyncio.to_thread(transcribe_audio, audio_path, system_prompt, task_id, model)
        
        # Удаляем аудиофайл сразу после транскрибации
        audio_path.unlink(missing_ok=True)
        logger.info(f"Аудиофайл удален для задачи {task_id}")
        
        # Обновляем статус: завершено
        task.status = "completed"
        task.stage = "completed"
        task.result = result
        task.updated_at = datetime.now()
        
    except Exception as e:
        # При ошибке удаляем все временные файлы
        video_path.unlink(missing_ok=True)
        if audio_path:
            audio_path.unlink(missing_ok=True)
        logger.info(f"Временные файлы удалены после ошибки для задачи {task_id}")
        
        if task_id in tasks:
            task = tasks[task_id]
            task.status = "error"
            task.error = str(e)
            task.updated_at = datetime.now()
        logger.error("Ошибка обработки задачи %s (%s)", task_id, type(e).__name__)


@app.get("/api/health")
async def healthcheck():
    return {
        "status": "ok",
        "ffmpeg": check_ffmpeg(),
        "api_key_configured": bool(DEFAULT_API_KEY),
        "folder_id_configured": bool(DEFAULT_FOLDER_ID),
    }


@app.get("/")
async def root():
    """Главная страница"""
    return FileResponse("static/index.html")


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None)
):
    """Загрузка видео файла для обработки"""
    
    # Проверяем наличие API ключей
    if not DEFAULT_API_KEY or not DEFAULT_FOLDER_ID:
        raise HTTPException(
            status_code=400,
            detail="API ключи не настроены. Установите YANDEX_API_KEY и YANDEX_FOLDER_ID в .env файле"
        )
    selected_model = LLM_MODEL if model is None else model
    try:
        validate_summary_model(selected_model)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    # Валидация формата
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не указано")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.mp4', '.webm']:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {file_ext}. Поддерживаются: .mp4, .webm"
        )
    
    # Создаем уникальный ID задачи
    task_id = str(uuid.uuid4())
    
    # Сохраняем файл
    video_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    
    try:
        content = await file.read()
        
        # Проверка размера
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Размер файла превышает 1GB")
        
        with open(video_path, 'wb') as f:
            f.write(content)
        
        # Проверка длительности видео
        try:
            duration = get_video_duration(video_path)
            if duration > MAX_VIDEO_DURATION:
                video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Длительность видео ({duration/60:.1f} мин) превышает 4 часа. Максимальная длительность: 240 минут"
                )
        except RuntimeError as e:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e))
        
    except HTTPException:
        raise
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {str(e)}")
    
    # Создаем запись о задаче
    task = TaskStatus(
        task_id=task_id,
        status="pending",
        stage="upload",
        filename=file.filename,
        model=selected_model,
    )
    tasks[task_id] = task
    
    # Запускаем фоновую обработку
    prompt = system_prompt or SYSTEM_PROMPT
    background_tasks.add_task(process_video_task, task_id, video_path, prompt, selected_model)
    
    return {
        "task_id": task_id,
        "message": "Файл загружен, обработка начата"
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса задачи"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = tasks[task_id]
    
    return {
        "task_id": task.task_id,
        "status": task.status,
        "stage": task.stage,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "filename": task.filename,
        "model": task.model,
    }


@app.get("/api/result/{task_id}")
async def get_task_result(task_id: str):
    """Получение результата задачи (задача удаляется после получения результата)"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Задача еще не завершена")
    
    if not task.result:
        raise HTTPException(status_code=404, detail="Результат не найден")
    
    result = task.result
    
    # Удаляем задачу после получения результата
    del tasks[task_id]
    logger.info(f"Задача {task_id} удалена после получения результата")
    
    return result


@app.get("/api/prompt")
async def get_system_prompt():
    """Получение промпта по умолчанию (обратная совместимость)"""
    return {"prompt": SYSTEM_PROMPT}


@app.get("/api/prompts")
async def get_prompts():
    """Список предустановленных промптов с текстом"""
    return {
        "prompts": [
            {"id": pid, "name": data["name"], "text": data["text"]}
            for pid, data in PRESET_PROMPTS.items()
        ],
        "default_id": "meeting",
    }




# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
