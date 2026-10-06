"""Регрессии обработки SpeechKit без сети, ffmpeg и настоящих ключей."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

with patch("dotenv.load_dotenv"), patch.dict(
    "os.environ",
    {
        "YANDEX_API_KEY": "test-key",
        "YANDEX_FOLDER_ID": "test-folder",
        "YANDEX_CLOUD_MODEL": "deepseek-v4.1-flash",
    },
):
    import app


def ndjson(*events):
    return "\n".join(json.dumps({"result": event}) for event in events)


def transcript(text="Обсудили тестовый проект."):
    return {"finalRefinement": {"normalizedText": {"alternatives": [{"text": text}]}}}


def summary(text="Договорились провести пилот."):
    return {"summarization": {"results": [{"response": text}]}}


class RecognitionResultTests(unittest.TestCase):
    def test_normalized_transcript_and_plain_summary(self):
        with self.assertLogs(app.logger, level="INFO") as logs:
            result = app.parse_recognition_result(ndjson(transcript(), summary()), "test-operation")
        self.assertEqual(result["transcription"], "Обсудили тестовый проект.")
        self.assertEqual(result["summary"], "Договорились провести пилот.")
        self.assertFalse(any(record.levelname == "ERROR" for record in logs.records))

    def test_json_summary(self):
        response = ndjson(summary('```json\n{"text": "Готовое резюме"}\n```'))
        self.assertEqual(app.parse_recognition_result(response, "test-operation")["summary"], "Готовое резюме")

    def test_unwrapped_event_and_blank_lines(self):
        response = "\n" + json.dumps(summary()) + "\n\n"
        self.assertEqual(app.parse_recognition_result(response, "test-operation")["summary"],
                         "Договорились провести пилот.")

    def test_missing_and_empty_summary_are_errors(self):
        for response in ("", ndjson(transcript()), ndjson(summary("")), ndjson(summary(" \n ")),
                         ndjson({"summarization": {"results": []}}), ndjson(summary(None))):
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, "не вернул резюме"):
                app.parse_recognition_result(response, "test-operation")

    def test_summary_empty_after_json_formatting_is_error(self):
        for text in ('{"text": ""}', '{"text": null}', '{"text": {}}', '{}'):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "пустой текст резюме"):
                app.parse_recognition_result(ndjson(summary(text)), "test-operation")

    def test_stream_error_after_valid_events_is_not_hidden(self):
        response = ndjson(transcript(), summary()) + "\n" + json.dumps({
            "error": {"code": 7, "message": "private transcript and test-key"},
        })
        with self.assertRaisesRegex(RuntimeError, "code=7, operation_id=test-operation") as raised:
            app.parse_recognition_result(response, "test-operation")
        self.assertNotIn("private transcript", str(raised.exception))
        self.assertNotIn("test-key", str(raised.exception))

    def test_malformed_json_is_not_silently_skipped(self):
        response = ndjson(summary()) + '\n{"result":'
        with self.assertRaisesRegex(RuntimeError, "JSON.*строка 2"):
            app.parse_recognition_result(response, "test-operation")

    def test_invalid_event_shape(self):
        for response in ('[]', 'null', '{"result": null}'):
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, "Некорректный формат"):
                app.parse_recognition_result(response, "test-operation")

    def test_warning_and_closed_do_not_override_valid_summary(self):
        response = ndjson(
            {"statusCode": {"codeType": "WARNING", "message": "private message"}},
            summary(),
            {"statusCode": {"codeType": "CLOSED"}},
        )
        with self.assertLogs(app.logger, level="INFO") as logs:
            result = app.parse_recognition_result(response, "test-operation")
        self.assertTrue(result["summary"])
        self.assertIn("WARNING", "\n".join(logs.output))
        self.assertNotIn("private message", "\n".join(logs.output))
        self.assertNotIn(result["summary"], "\n".join(logs.output))

    def test_closed_without_summary_is_error(self):
        response = ndjson(transcript(), {"statusCode": {"codeType": "CLOSED"}})
        with self.assertRaisesRegex(RuntimeError, "не вернул резюме"):
            app.parse_recognition_result(response, "test-operation")


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.audio = self.root / "test.mp3"
        self.audio.write_bytes(b"synthetic audio")
        self.tasks_patch = patch.object(app, "tasks", {})
        self.tasks_patch.start()
        self.addCleanup(self.tasks_patch.stop)

    def mock_speechkit(self, response_text):
        post = patch.object(app.requests, "post", return_value=Mock(
            status_code=200, json=lambda: {"id": "test-operation"},
        ))
        get = patch.object(app.requests, "get", side_effect=[
            Mock(status_code=200, json=lambda: {"done": True}),
            Mock(status_code=200, text=response_text),
        ])
        return post, get

    def test_transcribe_sends_model_and_selected_prompt(self):
        post, get = self.mock_speechkit(ndjson(transcript(), summary()))
        with post as request, get:
            result = app.transcribe_audio(self.audio, "Выбранный промпт", "test-task")
        self.assertEqual(result["summary"], "Договорились провести пилот.")
        self.assertEqual(request.call_args.kwargs["json"]["summarization"], {
            "modelUri": "gpt://test-folder/deepseek-v4.1-flash",
            "properties": [{"instruction": "Выбранный промпт"}],
        })

    def test_retired_model_is_rejected_before_network_request(self):
        for model in ("qwen3-235b-a22b-fp8", "qwen3-235b-a22b-fp8/latest"):
            with self.subTest(model=model), patch.object(app, "LLM_MODEL", model), \
                    patch.object(app.requests, "post") as request:
                with self.assertRaisesRegex(RuntimeError, "30 сентября 2026"):
                    app.transcribe_audio(self.audio, "Тестовый промпт")
                request.assert_not_called()

    def test_retired_model_is_rejected_at_upload(self):
        file = UploadFile(filename="test.mp4", file=io.BytesIO(b"synthetic video"))
        with patch.object(app, "LLM_MODEL", "qwen3-235b-a22b-fp8/latest"), \
                patch.object(app, "get_video_duration") as duration:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(app.upload_video(BackgroundTasks(), file, "Тестовый промпт", model=None))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("YANDEX_CLOUD_MODEL=deepseek-v4.1-flash", raised.exception.detail)
        self.assertEqual(file.file.tell(), 0)
        duration.assert_not_called()
        self.assertFalse(app.tasks)

    def test_unknown_model_is_rejected_before_saving_upload(self):
        for model in ("", "unknown-model", "gpt://another-folder/deepseek-v4.1-flash"):
            file = UploadFile(filename="test.mp4", file=io.BytesIO(b"synthetic video"))
            with self.subTest(model=model), patch.object(app, "get_video_duration") as duration:
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(app.upload_video(BackgroundTasks(), file, "Тестовый промпт", model=model))
                self.assertEqual(raised.exception.status_code, 400)
                self.assertIn("Неподдерживаемая модель", raised.exception.detail)
                self.assertEqual(file.file.tell(), 0)
                duration.assert_not_called()
        self.assertFalse(app.tasks)

    def test_each_upload_keeps_its_selected_model_through_background_processing(self):
        jobs = []
        # Даже старое значение в .env не переопределяет выбор в интерфейсе.
        with patch.object(app, "LLM_MODEL", "qwen3-235b-a22b-fp8/latest"), \
                patch.object(app, "UPLOAD_DIR", self.root), \
                patch.object(app, "get_video_duration", return_value=60):
            for model in ("deepseek-v4.1-flash", "qwen3.6-35b-a3b", "aliceai-llm-flash", "yandexgpt-5.1"):
                background = BackgroundTasks()
                file = UploadFile(filename="test.mp4", file=io.BytesIO(b"synthetic video"))
                response = asyncio.run(app.upload_video(background, file, "Выбранный промпт", model=model))
                jobs.append((background, response["task_id"], model))

            def extract(_video, audio, *_args):
                audio.write_bytes(b"synthetic audio")

            # Запускаем после создания всех задач и в обратном порядке, чтобы поймать общий mutable model.
            for background, task_id, model in reversed(jobs):
                post, get = self.mock_speechkit(ndjson(transcript(), summary()))
                with self.subTest(model=model), post as request, get, \
                        patch.object(app, "TEMP_DIR", self.root), \
                        patch.object(app, "extract_audio_from_video", side_effect=extract):
                    asyncio.run(background())
                self.assertEqual(request.call_args.kwargs["json"]["summarization"], {
                    "modelUri": f"gpt://test-folder/{model}",
                    "properties": [{"instruction": "Выбранный промпт"}],
                })
                status = asyncio.run(app.get_task_status(task_id))
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["model"], model)

    def test_api_without_model_keeps_configured_fallback(self):
        background = BackgroundTasks()
        file = UploadFile(filename="test.mp4", file=io.BytesIO(b"synthetic video"))
        with patch.object(app, "LLM_MODEL", "yandexgpt-5.1"), \
                patch.object(app, "UPLOAD_DIR", self.root), \
                patch.object(app, "get_video_duration", return_value=60):
            response = asyncio.run(app.upload_video(background, file, "Тестовый промпт", model=None))
        status = asyncio.run(app.get_task_status(response["task_id"]))
        self.assertEqual(status["model"], "yandexgpt-5.1")

    def run_video_task(self, response_text):
        video = self.root / "test.mp4"
        video.write_bytes(b"synthetic video")
        app.tasks["test-task"] = app.TaskStatus("test-task", "pending", "upload")

        def extract(_video, audio, *_args):
            audio.write_bytes(b"synthetic audio")

        post, get = self.mock_speechkit(response_text)
        with post, get, patch.object(app, "TEMP_DIR", self.root), \
                patch.object(app, "extract_audio_from_video", side_effect=extract):
            asyncio.run(app.process_video_task("test-task", video, "Тестовый промпт"))
        self.assertFalse(video.exists())
        self.assertFalse((self.root / "test-task.mp3").exists())
        return asyncio.run(app.get_task_status("test-task"))

    def test_missing_summary_sets_task_error(self):
        status = self.run_video_task(ndjson(transcript()))
        self.assertEqual(status["status"], "error")
        self.assertIsNone(status["result"])
        self.assertIn("не вернул резюме", status["error"])
        self.assertIn("test-operation", status["error"])
        with self.assertRaises(HTTPException):
            asyncio.run(app.get_task_result("test-task"))

    def test_summary_completes_task_and_result_can_be_retrieved(self):
        status = self.run_video_task(ndjson(transcript(), summary()))
        self.assertEqual(status["status"], "completed")
        result = asyncio.run(app.get_task_result("test-task"))
        self.assertEqual(result["summary"], "Договорились провести пилот.")
        self.assertNotIn("test-task", app.tasks)


if __name__ == "__main__":
    unittest.main()
