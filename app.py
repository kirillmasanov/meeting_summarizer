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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Cookie, Response, Request
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
DEFAULT_FOLDER_ID = os.getenv("FOLDER_ID")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-235b-a22b-fp8/latest")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
MAX_VIDEO_DURATION = 14400  # 4 часа в секундах
UPLOAD_TIMEOUT = 600  # 10 минут
STATUS_CHECK_TIMEOUT = 30  # 30 секунд
RESULT_TIMEOUT = 60  # 1 минута
POLL_INTERVAL = 5  # 5 секунд между проверками статуса
SESSION_TTL = timedelta(minutes=30)  # TTL сессии

# Логируем наличие дефолтных ключей
if DEFAULT_API_KEY and DEFAULT_FOLDER_ID:
    logger.info("Найдены дефолтные API ключи в переменных окружения")
else:
    logger.info("Дефолтные API ключи не найдены, потребуется ввод через UI")

# Загружаем системный промпт из файла
prompt_file = Path("system_prompt.txt")
if not prompt_file.exists():
    raise FileNotFoundError(f"Файл {prompt_file} не найден. Создайте файл с системным промптом.")

with open(prompt_file, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read().strip()

# Создаем FastAPI приложение
app = FastAPI(title="Meeting Summarizer", version="1.0.0")

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


# Модели данных для сессий
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


@dataclass
class SessionData:
    """Данные пользовательской сессии"""
    session_id: str
    api_key: str
    folder_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    tasks: Dict[str, TaskStatus] = field(default_factory=dict)


# Хранилище сессий
sessions: Dict[str, SessionData] = {}


def cleanup_expired_sessions():
    """Удаляет неактивные сессии"""
    now = datetime.now()
    expired = [
        sid for sid, session in sessions.items()
        if now - session.last_activity > SESSION_TTL
    ]
    for sid in expired:
        logger.info(f"Удаление неактивной сессии: {sid}")
        del sessions[sid]


def get_or_create_session(session_id: Optional[str], api_key: Optional[str] = None, folder_id: Optional[str] = None) -> SessionData:
    """Получает существующую сессию или создает новую"""
    cleanup_expired_sessions()
    
    # Если session_id не передан, создаем новую сессию
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Создание новой сессии: {session_id}")
    
    # Если сессия существует, обновляем время активности
    if session_id in sessions:
        session = sessions[session_id]
        session.last_activity = datetime.now()
        
        # Обновляем ключи если переданы новые
        if api_key:
            session.api_key = api_key
        if folder_id:
            session.folder_id = folder_id
            
        return session
    
    # Создаем новую сессию
    # Используем переданные ключи или дефолтные из env
    final_api_key = api_key or DEFAULT_API_KEY
    final_folder_id = folder_id or DEFAULT_FOLDER_ID
    
    if not final_api_key or not final_folder_id:
        raise ValueError("API ключи не предоставлены и не найдены в переменных окружения")
    
    session = SessionData(
        session_id=session_id,
        api_key=final_api_key,
        folder_id=final_folder_id
    )
    sessions[session_id] = session
    logger.info(f"Создана новая сессия: {session_id}")
    
    return session


def get_session(session_id: str) -> SessionData:
    """Получает существующую сессию или выбрасывает ошибку"""
    cleanup_expired_sessions()
    
    if session_id not in sessions:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")
    
    session = sessions[session_id]
    session.last_activity = datetime.now()
    return session


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


def update_task_timestamp(session_id: str, task_id: str):
    """Обновляет временную метку задачи"""
    if session_id in sessions and task_id in sessions[session_id].tasks:
        sessions[session_id].tasks[task_id].updated_at = datetime.now()


def transcribe_audio(audio_file_path: Path, system_prompt: str, session: SessionData, task_id: str = None) -> dict:
    """Транскрибирует аудио файл через Yandex SpeechKit API v3"""
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
            "modelUri": f"gpt://{session.folder_id}/{LLM_MODEL}",
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
        "Authorization": f"Api-key {session.api_key}"
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
            update_task_timestamp(session.session_id, task_id)
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
    
    # Парсим результаты
    results = speech_response.text.strip().split('\n')
    transcription_parts = []
    summary = None
    
    for line_num, line in enumerate(results):
        try:
            data = json.loads(line)
            result = data.get("result", {})
            
            # Извлекаем транскрипцию
            if "finalRefinement" in result:
                normalized = result["finalRefinement"].get("normalizedText", {})
                alternatives = normalized.get("alternatives", [])
                if alternatives:
                    text = alternatives[0].get("text", "")
                    if text:
                        transcription_parts.append(text)
            
            # Извлекаем суммаризацию
            if "summarization" in result:
                summarization = result["summarization"]
                results_list = summarization.get("results", [])
                if results_list:
                    raw_summary = results_list[0].get("response", "")
                    if raw_summary and raw_summary.strip():
                        summary = raw_summary.strip()
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON в строке {line_num}: {e}")
            continue
    
    full_transcription = " ".join(transcription_parts)

    # Конвертируем JSON в текст если нужно
    if summary:
        summary_html = json_to_html(summary)
    else:
        summary_html = None
        logger.warning("Summary пустой, будет использовано 'Резюме не создано'")
    
    return {
        "transcription": full_transcription,
        "summary": summary_html or "Резюме не создано"
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
            
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        return text


async def process_video_task(session_id: str, task_id: str, video_path: Path, system_prompt: str):
    """Фоновая задача обработки видео"""
    audio_path = None
    try:
        session = sessions.get(session_id)
        if not session:
            logger.error(f"Сессия {session_id} не найдена при обработке задачи {task_id}")
            return
        
        # Обновляем статус: извлечение аудио
        task = session.tasks[task_id]
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
        result = await asyncio.to_thread(transcribe_audio, audio_path, system_prompt, session, task_id)
        
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
        
        if session_id in sessions and task_id in sessions[session_id].tasks:
            task = sessions[session_id].tasks[task_id]
            task.status = "error"
            task.error = str(e)
            task.updated_at = datetime.now()


@app.get("/")
async def root():
    """Главная страница"""
    return FileResponse("static/index.html")


@app.post("/api/validate-keys")
async def validate_api_keys(
    api_key: str = Form(...),
    folder_id: str = Form(...)
):
    """Валидация API ключей через тестовый запрос к Yandex Cloud"""
    try:
        # Проверяем формат ключей
        if not api_key or len(api_key) < 20:
            raise HTTPException(status_code=400, detail="Некорректный формат API ключа")
        
        if not folder_id or not folder_id.startswith('b1'):
            raise HTTPException(status_code=400, detail="Некорректный формат Folder ID (должен начинаться с 'b1')")
        
        # Делаем тестовый запрос к Yandex Cloud API для проверки ключей
        headers = {
            "Authorization": f"Api-key {api_key}"
        }
        
        # Используем простой endpoint для проверки авторизации
        test_url = f"https://operation.api.cloud.yandex.net/operations"
        
        try:
            response = requests.get(
                test_url,
                headers=headers,
                verify=False,
                timeout=10
            )
            
            # Если получили 401 или 403 - ключ невалидный
            if response.status_code in [401, 403]:
                raise HTTPException(status_code=400, detail="API ключ недействителен или не имеет необходимых прав")
            
            # Любой другой ответ (включая 200, 400, 404) означает что ключ валиден
            logger.info(f"API ключи валидированы успешно (status: {response.status_code})")
            
        except requests.exceptions.Timeout:
            # Таймаут не означает невалидный ключ, просто не можем проверить
            logger.warning("Таймаут при валидации ключей, пропускаем проверку")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка при валидации ключей: {e}, пропускаем проверку")
        
        return {
            "valid": True,
            "message": "API ключи валидны"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка валидации ключей: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при валидации ключей")


@app.post("/api/session")
async def create_session(
    response: Response,
    api_key: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None)
):
    """Создание новой сессии с опциональными API ключами"""
    try:
        session = get_or_create_session(None, api_key, folder_id)
        
        # Устанавливаем cookie с session_id
        response.set_cookie(
            key="session_id",
            value=session.session_id,
            httponly=True,
            max_age=1800,  # 30 минут
            samesite="lax"
        )
        
        return {
            "session_id": session.session_id,
            "message": "Сессия создана",
            "has_default_keys": bool(DEFAULT_API_KEY and DEFAULT_FOLDER_ID)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/session/check")
async def check_session(session_id: Optional[str] = Cookie(None)):
    """Проверка наличия активной сессии"""
    if not session_id or session_id not in sessions:
        return {
            "has_session": False,
            "has_default_keys": bool(DEFAULT_API_KEY and DEFAULT_FOLDER_ID),
            "requires_keys": not (DEFAULT_API_KEY and DEFAULT_FOLDER_ID)
        }
    
    session = sessions[session_id]
    session.last_activity = datetime.now()
    
    return {
        "has_session": True,
        "session_id": session_id,
        "has_default_keys": bool(DEFAULT_API_KEY and DEFAULT_FOLDER_ID),
        "requires_keys": False
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    response: Response,
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Загрузка видео файла для обработки"""
    
    # Получаем или создаем сессию
    try:
        session = get_or_create_session(session_id, api_key, folder_id)
        
        # Обновляем cookie если сессия новая
        if not session_id or session_id != session.session_id:
            response.set_cookie(
                key="session_id",
                value=session.session_id,
                httponly=True,
                max_age=1800,
                samesite="lax"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
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
    
    # Создаем запись о задаче в сессии
    task = TaskStatus(
        task_id=task_id,
        status="pending",
        stage="upload",
        filename=file.filename
    )
    session.tasks[task_id] = task
    
    # Запускаем фоновую обработку
    prompt = system_prompt or SYSTEM_PROMPT
    background_tasks.add_task(process_video_task, session.session_id, task_id, video_path, prompt)
    
    return {
        "task_id": task_id,
        "session_id": session.session_id,
        "message": "Файл загружен, обработка начата"
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str, session_id: Optional[str] = Cookie(None)):
    """Получение статуса задачи"""
    if not session_id:
        raise HTTPException(status_code=401, detail="Сессия не найдена")
    
    session = get_session(session_id)
    
    if task_id not in session.tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = session.tasks[task_id]
    
    return {
        "task_id": task.task_id,
        "status": task.status,
        "stage": task.stage,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "filename": task.filename
    }


@app.get("/api/result/{task_id}")
async def get_task_result(task_id: str, session_id: Optional[str] = Cookie(None)):
    """Получение результата задачи (задача удаляется после получения результата)"""
    if not session_id:
        raise HTTPException(status_code=401, detail="Сессия не найдена")
    
    session = get_session(session_id)
    
    if task_id not in session.tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = session.tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Задача еще не завершена")
    
    if not task.result:
        raise HTTPException(status_code=404, detail="Результат не найден")
    
    result = task.result
    
    # Удаляем задачу из сессии после получения результата
    del session.tasks[task_id]
    logger.info(f"Задача {task_id} удалена из сессии {session_id} после получения результата")
    
    return result


@app.get("/api/prompt")
async def get_system_prompt():
    """Получение текущего системного промпта"""
    return {"prompt": SYSTEM_PROMPT}


@app.on_event("startup")
async def startup_event():
    """Запуск фоновых задач при старте приложения"""
    # Проверяем наличие ffmpeg
    if not check_ffmpeg():
        logger.warning("ffmpeg не найден в системе!")
        logger.warning("Установите ffmpeg для работы приложения")


# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)