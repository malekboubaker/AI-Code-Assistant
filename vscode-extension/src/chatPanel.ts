import * as vscode from 'vscode';
import { GenerateResponse } from './apiClient';

export type ChatViewMessage =
  | { type: 'send'; text: string }
  | { type: 'apply'; responseId: string }
  | { type: 'checkRagStatus' }
  | { type: 'indexProject'; mode: 'incremental' | 'full' }
  | { type: 'resetProjectIndex' };

export interface AssistantResponsePayload {
  responseId: string;
  instruction: string;
  response: GenerateResponse;
  contextSummary: string;
  showTiming: boolean;
}

export function getChatViewHtml(webview: vscode.Webview): string {
  const nonce = getNonce();
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta
    http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';"
  >
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Code Assistant</title>
  <style nonce="${nonce}">
    :root {
      color-scheme: light dark;
      --border: var(--vscode-panel-border);
      --muted: var(--vscode-descriptionForeground);
      --error: var(--vscode-errorForeground);
    }

    body {
      margin: 0;
      padding: 0;
      color: var(--vscode-foreground);
      background: var(--vscode-sideBar-background);
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
    }

    .shell {
      display: flex;
      flex-direction: column;
      height: 100vh;
      min-height: 0;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
    }

    .rag-status-panel {
      border-bottom: 1px solid var(--border);
      padding: 8px 12px;
      background: var(--vscode-sideBar-background);
    }

    .rag-status-title {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    .rag-status-body {
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .rag-status-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .message {
      margin-bottom: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .message-title {
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }

    .bubble {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      background: var(--vscode-editor-background);
      overflow-wrap: anywhere;
    }

    .user .bubble {
      background: var(--vscode-input-background);
    }

    .error .bubble {
      color: var(--error);
      border-color: var(--error);
    }

    .composer {
      border-top: 1px solid var(--border);
      padding: 8px;
      background: var(--vscode-sideBar-background);
    }

    textarea {
      width: 100%;
      min-height: 72px;
      max-height: 180px;
      box-sizing: border-box;
      resize: vertical;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
      border: 1px solid var(--vscode-input-border);
      border-radius: 4px;
      padding: 8px;
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }

    button {
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 4px;
      padding: 5px 10px;
      color: var(--vscode-button-foreground);
      background: var(--vscode-button-background);
      cursor: pointer;
    }

    button:hover {
      background: var(--vscode-button-hoverBackground);
    }

    button.secondary {
      color: var(--vscode-button-secondaryForeground);
      background: var(--vscode-button-secondaryBackground);
    }

    button.secondary:hover {
      background: var(--vscode-button-secondaryHoverBackground);
    }

    button:disabled {
      opacity: 0.55;
      cursor: default;
    }

    .status {
      color: var(--muted);
      font-size: 12px;
    }

    pre {
      margin: 8px 0 0;
      padding: 8px;
      overflow: auto;
      white-space: pre;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: var(--vscode-textCodeBlock-background);
    }

    code {
      font-family: var(--vscode-editor-font-family);
      font-size: var(--vscode-editor-font-size);
    }

    details {
      margin-top: 8px;
    }

    summary {
      cursor: pointer;
      color: var(--muted);
    }

    .source {
      margin-top: 4px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }

    .sources-section {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }

    .sources-title {
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }

    .source-item {
      margin-top: 6px;
      color: var(--vscode-foreground);
      overflow-wrap: anywhere;
    }

    .source-meta {
      color: var(--muted);
      font-size: 12px;
    }

    .empty-state {
      color: var(--muted);
      padding: 12px;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div id="ragStatusPanel" class="rag-status-panel">
      <div class="rag-status-title">RAG index</div>
      <div id="ragStatusBody" class="rag-status-body">Checking workspace index...</div>
      <div class="rag-status-actions">
        <button id="checkRagStatus" class="secondary">Check RAG Status</button>
        <button id="indexProject" class="secondary">Index Project</button>
        <button id="fullIndexProject" class="secondary">Full Re-index Project</button>
        <button id="resetProjectIndex" class="secondary">Reset Project Index</button>
      </div>
    </div>
    <div id="messages" class="messages" aria-live="polite">
      <div class="empty-state">Ask about the current selection, active file, or indexed workspace.</div>
    </div>
    <div class="composer">
      <textarea id="input" placeholder="Ask: Optimize this function, explain this file, fix the API timeout issue..."></textarea>
      <div class="actions">
        <button id="send">Send</button>
        <span id="status" class="status"></span>
      </div>
    </div>
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const messages = document.getElementById('messages');
    const input = document.getElementById('input');
    const send = document.getElementById('send');
    const status = document.getElementById('status');
    const ragStatusBody = document.getElementById('ragStatusBody');
    const checkRagStatus = document.getElementById('checkRagStatus');
    const indexProject = document.getElementById('indexProject');
    const fullIndexProject = document.getElementById('fullIndexProject');
    const resetProjectIndex = document.getElementById('resetProjectIndex');

    function clearEmptyState() {
      const empty = messages.querySelector('.empty-state');
      if (empty) {
        empty.remove();
      }
    }

    function appendMessage(kind, title, body) {
      clearEmptyState();
      const wrapper = document.createElement('div');
      wrapper.className = 'message ' + kind;
      const heading = document.createElement('div');
      heading.className = 'message-title';
      heading.textContent = title;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = body;
      wrapper.appendChild(heading);
      wrapper.appendChild(bubble);
      messages.appendChild(wrapper);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    }

    function appendAssistant(payload) {
      const response = payload.response;
      const hasCode = Boolean(response.generated_code && response.generated_code.trim());
      const hasExplanation = Boolean(response.explanation && response.explanation.trim());
      const bubble = appendMessage(
        'assistant',
        response.task ? 'Assistant - ' + response.task : 'Assistant',
        hasExplanation ? response.explanation : (hasCode ? 'Generated code suggestion:' : 'No response text returned.')
      );

      const context = document.createElement('div');
      context.className = 'source';
      context.textContent = payload.contextSummary;
      bubble.appendChild(context);

      if (hasCode) {
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.textContent = response.generated_code;
        pre.appendChild(code);
        bubble.appendChild(pre);

        const apply = document.createElement('button');
        apply.className = 'secondary';
        apply.textContent = 'Apply';
        apply.addEventListener('click', () => {
          vscode.postMessage({ type: 'apply', responseId: payload.responseId });
        });
        const actionRow = document.createElement('div');
        actionRow.className = 'actions';
        actionRow.appendChild(apply);
        bubble.appendChild(actionRow);
      }

      if (response.diff) {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Diff preview';
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.textContent = response.diff;
        pre.appendChild(code);
        details.appendChild(summary);
        details.appendChild(pre);
        bubble.appendChild(details);
      }

      if (Array.isArray(response.rag_sources) && response.rag_sources.length > 0) {
        appendSourcesSection(bubble, response.rag_sources);
      }

      if (payload.showTiming && response.metadata) {
        const timing = {
          total_ms: response.metadata.timing_total_ms,
          rag_ms: response.metadata.timing_rag_ms,
          model_ms: response.metadata.timing_model_ms,
          validation_ms: response.metadata.timing_validation_ms,
          model: response.metadata.model_name,
          prompt_chars: response.metadata.prompt_length_chars,
          generated_chars: response.metadata.generated_length_chars
        };
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Timing metadata';
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.textContent = JSON.stringify(timing, null, 2);
        pre.appendChild(code);
        details.appendChild(summary);
        details.appendChild(pre);
        bubble.appendChild(details);
      }

      messages.scrollTop = messages.scrollHeight;
    }

    function appendSourcesMessage(sources, message) {
      const bubble = appendMessage('assistant', 'Sources used', message || 'Sources used for the previous response.');
      if (Array.isArray(sources) && sources.length > 0) {
        appendSourcesSection(bubble, sources);
      }
    }

    function appendSourcesSection(container, sources) {
      const section = document.createElement('div');
      section.className = 'sources-section';
      const title = document.createElement('div');
      title.className = 'sources-title';
      title.textContent = 'Sources used';
      section.appendChild(title);

      for (const source of sources) {
        const item = document.createElement('div');
        item.className = 'source-item';

        const primary = document.createElement('div');
        primary.textContent = '- ' + sourceDisplayPath(source);
        item.appendChild(primary);

        const meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.textContent = sourceMetadataText(source);
        item.appendChild(meta);

        section.appendChild(item);
      }

      container.appendChild(section);
    }

    function sourceDisplayPath(source) {
      const metadata = source && source.metadata && typeof source.metadata === 'object' ? source.metadata : {};
      return String(metadata.relative_file_path || metadata.relative_path || source.file_path || 'unknown');
    }

    function sourceMetadataText(source) {
      const metadata = source && source.metadata && typeof source.metadata === 'object' ? source.metadata : {};
      const parts = [];
      if (source.file_path) {
        parts.push('file_path=' + source.file_path);
      }
      if (source.symbol_name) {
        parts.push('symbol=' + source.symbol_name);
      }
      if (source.chunk_type) {
        parts.push('type=' + source.chunk_type);
      }
      if (source.start_line || source.end_line) {
        parts.push('lines=' + (source.start_line || '?') + '-' + (source.end_line || '?'));
      }
      if (typeof source.score === 'number') {
        parts.push('score=' + source.score.toFixed(3));
      }
      if (metadata.project_id) {
        parts.push('project_id=' + metadata.project_id);
      }
      return parts.join(' | ');
    }

    function renderRagStatus(payload) {
      if (!ragStatusBody) {
        return;
      }
      if (payload.error) {
        ragStatusBody.textContent = payload.error;
        return;
      }
      const status = payload.status;
      if (!status) {
        ragStatusBody.textContent = payload.message || 'RAG status unavailable.';
        return;
      }
      const projectName = status.project_path ? String(status.project_path).split(/[\\\\/]/).filter(Boolean).pop() : 'Workspace';
      const state = status.indexed && status.project_map_exists ? 'Indexed' : 'RAG index not ready';
      const qdrant = status.qdrant_ready === false ? ' | Qdrant offline or unavailable' : '';
      const lines = [
        projectName + ': ' + state + qdrant,
        'Project ID: ' + status.project_id,
        'Points: ' + status.point_count + ' | Project map: ' + (status.project_map_exists ? 'yes' : 'no'),
        'Last indexed: ' + (status.last_indexed || 'unknown')
      ];
      if (Array.isArray(status.frameworks) && status.frameworks.length > 0) {
        lines.push('Frameworks: ' + status.frameworks.join(', '));
      }
      if (Array.isArray(status.entry_points) && status.entry_points.length > 0) {
        lines.push('Entry points: ' + status.entry_points.slice(0, 3).join(', '));
      }
      if (payload.message) {
        lines.push(payload.message);
      }
      ragStatusBody.textContent = lines.join('\\n');
    }

    function renderRagIndexing(message) {
      if (ragStatusBody) {
        ragStatusBody.textContent = message;
      }
    }

    function setLoading(isLoading) {
      send.disabled = isLoading;
      status.textContent = isLoading ? 'Thinking locally...' : '';
    }

    function submit() {
      const text = input.value.trim();
      if (!text) {
        return;
      }
      appendMessage('user', 'You', text);
      input.value = '';
      setLoading(true);
      vscode.postMessage({ type: 'send', text });
    }

    send.addEventListener('click', submit);
    checkRagStatus.addEventListener('click', () => {
      vscode.postMessage({ type: 'checkRagStatus' });
    });
    indexProject.addEventListener('click', () => {
      vscode.postMessage({ type: 'indexProject', mode: 'incremental' });
    });
    fullIndexProject.addEventListener('click', () => {
      vscode.postMessage({ type: 'indexProject', mode: 'full' });
    });
    resetProjectIndex.addEventListener('click', () => {
      vscode.postMessage({ type: 'resetProjectIndex' });
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        submit();
      }
    });

    window.addEventListener('message', (event) => {
      const message = event.data;
      if (message.type === 'assistant') {
        setLoading(false);
        appendAssistant(message.payload);
      } else if (message.type === 'sources') {
        setLoading(false);
        appendSourcesMessage(message.sources, message.message);
      } else if (message.type === 'ragStatus') {
        renderRagStatus(message.payload);
      } else if (message.type === 'ragIndexing') {
        renderRagIndexing(message.message);
      } else if (message.type === 'error') {
        setLoading(false);
        appendMessage('error', 'Error', message.message);
      } else if (message.type === 'applied') {
        appendMessage('assistant', 'Applied', message.message);
      }
    });
  </script>
</body>
</html>`;
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let nonce = '';
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}
