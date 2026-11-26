import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createContext, runInContext } from 'node:vm';

const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');

function harness({ native, preview = false } = {}) {
  const elements = new Map();
  const status = { dataset: {}, title: '' };
  const context = createContext({
    console, URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval,
    localStorage: { getItem: () => null },
    window: { location: { search: preview ? '?preview=1' : '' }, addEventListener() {},
      ...(native ? { __TAURI__: { core: { invoke: native } } } : {}) },
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, { textContent: '', style: {}, innerHTML: '',
          closest: () => status, appendChild() {} });
        return elements.get(id);
      },
      createElement(tag) {
        return {
          tagName: tag.toUpperCase(),
          className: '',
          textContent: '',
          innerHTML: '',
          style: {},
          appendChild() {},
        };
      }
    },
  });
  runInContext(source, context);
  runInContext('showToast = () => {}', context);
  return { context, elements, status, run: code => runInContext(code, context) };
}

test('ordinary browser mode cannot silently return mock chat', async () => {
  const app = harness();
  await assert.rejects(app.run("invoke('ask_robbie', {prompt: 'hello'})"), /launch_editor/);
});

test('preview mode is explicit and identifies sample data', async () => {
  const app = harness({ preview: true });
  await app.run('syncBackendStatus()');
  assert.equal(app.elements.get('sidebar-model-label').textContent, 'Preview — sample data');
  assert.equal(app.status.dataset.connected, 'false');
});

test('backend errors propagate instead of being treated as an empty answer', async () => {
  const app = harness({ native: async () => JSON.stringify({ error: 'Missing checkpoint' }) });
  await assert.rejects(app.run("invoke('ask_robbie', {prompt: 'hello'})"), /Missing checkpoint/);
});

test('status displays the backend checkpoint and actual configuration', async () => {
  const app = harness({ native: async () => JSON.stringify({ loaded: true, model: 'Robbie 26.4M',
    device: 'cuda', checkpoint: 'checkpoints/answers_run/best_answers.pt',
    context_tokens: 256, max_tokens: 100, temperature: 0 }) });
  await app.run('syncBackendStatus()');
  assert.equal(app.elements.get('sidebar-model-label').textContent, 'Robbie 26.4M · CUDA');
  assert.match(app.elements.get('settings-model-checkpoint').textContent, /answers_run/);
  assert.match(app.elements.get('settings-model-config').textContent, /256 token context/);
  assert.equal(app.status.dataset.connected, 'true');
});

test('new chat resets backend before clearing the visible conversation', async () => {
  const calls = [];
  const app = harness({ native: async command => { calls.push(command); return '{"turns":0}'; } });
  app.run("state.chatMessages = [{role: 'user', content: 'hello'}]");
  await app.run('resetChatSession()');
  assert.deepEqual(calls, ['reset_chat']);
  assert.equal(app.run('state.chatMessages.length'), 0);
});

test('failed reset preserves the visible conversation', async () => {
  const app = harness({ native: async () => { throw new Error('Disconnected'); } });
  app.run("state.chatMessages = [{role: 'user', content: 'hello'}]");
  await app.run('resetChatSession()');
  assert.equal(app.run('state.chatMessages.length'), 1);
});

test('chat history restoration correctly identifies Robbie assistant vs user messages', async () => {
  const app = harness({ native: async () => JSON.stringify({
    loaded: true,
    model: 'Robbie 50M',
    device: 'cuda',
    checkpoint: 'checkpoints/stage2_50m_conversational_transformer.pt',
    context_tokens: 512,
    max_tokens: 128,
    temperature: 0.7,
    turns: 2,
    history: [
      { role: 'User', content: 'hello from human' },
      { role: 'Robbie', content: 'Hello! I am Robbie.' }
    ]
  }) });
  await app.run('syncBackendStatus()');
  assert.equal(app.run('state.chatMessages.length'), 2);
  assert.equal(app.run('state.chatMessages[0].role'), 'user');
  assert.equal(app.run('state.chatMessages[1].role'), 'assistant');
});
