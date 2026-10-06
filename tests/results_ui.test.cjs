const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(new URL('../static/index.html', `file://${__filename}`), 'utf8');
const showResults = html.slice(html.indexOf('async function showResults()'), html.indexOf('// Копирование резюме'));
const pollStatus = html.slice(html.indexOf('async function pollTaskStatus()'), html.indexOf('// Обновление прогресса'));
const upload = html.slice(html.indexOf("uploadButton.addEventListener('click'"), html.indexOf('// Отслеживание статуса задачи'));

function context(response, data) {
    const state = { errors: [], visible: false, reset: false, polls: 0 };
    const scope = vm.createContext({
        API_BASE: '', currentTaskId: 'test-task',
        fetch: async () => ({ ...response, json: async () => data }),
        summaryText: { textContent: '' },
        progressSection: { classList: { remove() {} } },
        resultsSection: { classList: { add() { state.visible = true; } } },
        showError(message) { state.errors.push(message); },
        resetToUpload() { state.reset = true; },
        updateProgress() {},
        setTimeout() { state.polls += 1; },
    });
    vm.runInContext(`${showResults}\n${pollStatus}`, scope);
    return { state, scope };
}

test('successful response renders the summary', async () => {
    const { state, scope } = context({ ok: true }, { summary: 'Готовое резюме' });
    await scope.showResults();
    assert.equal(scope.summaryText.textContent, 'Готовое резюме');
    assert.equal(state.visible, true);
    assert.deepEqual(state.errors, []);
});

test('HTTP errors and empty results do not show a successful summary', async () => {
    for (const [response, data, expected] of [
        [{ ok: false, status: 404 }, { detail: 'Задача не найдена' }, 'Задача не найдена'],
        [{ ok: false, status: 500 }, {}, 'Ошибка сервера (500)'],
        [{ ok: true }, { summary: '  ' }, 'пустое резюме'],
        [{ ok: true }, { summary: null }, 'пустое резюме'],
        [{ ok: true }, {}, 'пустое резюме'],
    ]) {
        const { state, scope } = context(response, data);
        await scope.showResults();
        assert.equal(state.visible, false);
        assert.equal(state.reset, true);
        assert.ok(state.errors[0].includes(expected));
    }
});

test('a failed task displays its error and stops polling', async () => {
    const { state, scope } = context({ ok: true }, { status: 'error', error: 'SpeechKit не вернул резюме' });
    await scope.pollTaskStatus();
    assert.deepEqual(state.errors, ['SpeechKit не вернул резюме']);
    assert.equal(state.polls, 0);
    assert.equal(state.visible, false);
    assert.equal(state.reset, true);
});

test('a missing task stops polling instead of waiting forever', async () => {
    const { state, scope } = context({ ok: false, status: 404 }, { detail: 'Задача не найдена' });
    await scope.pollTaskStatus();
    assert.ok(state.errors[0].includes('Задача не найдена'));
    assert.equal(state.polls, 0);
    assert.equal(state.reset, true);
});

test('upload sends the selected model and current prompt with the file', async () => {
    for (const model of ['deepseek-v4.1-flash', 'qwen3.6-35b-a3b', 'aliceai-llm-flash', 'yandexgpt-5.1']) {
        let handler;
        let sent;
        const errors = [];
        const scope = vm.createContext({
            API_BASE: '', currentTaskId: null,
            selectedFile: new Blob(['synthetic video'], { type: 'video/mp4' }),
            currentPrompt: 'Выбранный промпт', summaryModel: { value: model }, FormData,
            uploadButton: { addEventListener(_event, callback) { handler = callback; } },
            uploadSection: { style: {} }, progressSection: { classList: { add() {} } },
            pollTaskStatus() {}, showError(message) { errors.push(message); },
            fetch: async (url, options) => {
                sent = { url, ...options };
                return {
                    ok: true, headers: { get: () => 'application/json' },
                    json: async () => ({ task_id: 'test-task' }),
                };
            },
        });
        vm.runInContext(upload, scope);
        await handler();
        assert.deepEqual(errors, []);
        assert.equal(sent.url, '/api/upload');
        assert.equal(sent.method, 'POST');
        assert.equal(sent.body.get('model'), model);
        assert.equal(sent.body.get('system_prompt'), 'Выбранный промпт');
        assert.equal(await sent.body.get('file').text(), 'synthetic video');
    }
});
