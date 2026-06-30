import * as vscode from 'vscode';
import { GenerateResponse } from './apiClient';

export type ChatViewMessage =
  | { type: 'send'; text: string }
  | { type: 'apply'; responseId: string }
  | { type: 'openDiff'; responseId: string }
  | { type: 'openFile'; filePath: string; line?: number }
  | { type: 'copyText'; text: string }
  | { type: 'newConversation' }
  | { type: 'generateReport' }
  | { type: 'openReport' }
  | { type: 'checkRagStatus' }
  | { type: 'indexProject'; mode: 'incremental' | 'full' }
  | { type: 'resetProjectIndex' }
  | { type: 'applyAll'; responseId: string }
  | { type: 'stopGeneration' }
  | { type: 'webviewError'; message: string }
  | { type: 'requestEditorState' }
  | { type: 'openReport' }
  | { type: 'copyText'; text: string }
  | { type: 'openDiff'; responseId: string }
  | { type: 'openFile'; filePath: string; line?: number }
  | { type: 'regenerate'; responseId: string; text: string };

export interface AssistantResponsePayload {
  responseId: string;
  instruction: string;
  response: GenerateResponse;
  contextSummary: string;
  showTiming: boolean;
}

export function getChatViewHtml(webview: vscode.Webview): string {
  const nonce = getNonce();
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta
    http-equiv="Content-Security-Policy"
    content="default-src 'none'; img-src ${webview.cspSource} data:; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';"
  >
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Code Assistant</title>
  <style nonce="${nonce}">
    :root {
      color-scheme: light dark;
      --border: var(--vscode-panel-border, rgba(128,128,128,0.25));
      --muted: var(--vscode-descriptionForeground);
      --error: var(--vscode-errorForeground);
      --accent: var(--vscode-textLink-foreground);
      --radius: 8px;
      --tok-keyword: var(--vscode-symbolIcon-keywordForeground, #569cd6);
      --tok-string: var(--vscode-debugTokenExpression-string, #ce9178);
      --tok-number: var(--vscode-debugTokenExpression-number, #b5cea8);
      --tok-comment: var(--vscode-descriptionForeground, #6a9955);
      --tok-function: var(--vscode-symbolIcon-functionForeground, #dcdcaa);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 0;
      color: var(--vscode-foreground);
      background: var(--vscode-sideBar-background);
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
    }

    .app { display: flex; flex-direction: column; height: 100vh; min-height: 0; }

    /* Header */
    .header {
      border-bottom: 1px solid var(--border);
      padding: 8px 12px;
      background: var(--vscode-sideBar-background);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .header-top { display: flex; align-items: center; gap: 8px; }
    .project { display: flex; align-items: center; gap: 6px; font-weight: 600; min-width: 0; }
    .project span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge {
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--muted);
      white-space: nowrap;
    }
    .badge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
    .badge.ok { color: var(--vscode-testing-iconPassed, #3fb950); border-color: currentColor; }
    .badge.ok .dot { background: currentColor; }
    .badge.warn { color: var(--vscode-editorWarning-foreground, #d7972a); border-color: currentColor; }
    .badge.warn .dot { background: currentColor; }
    .badge.busy { color: var(--accent); border-color: currentColor; }
    .badge.busy .dot { background: currentColor; animation: pulse 1s infinite; }

    .header-file {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }
    .header-file .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sel-indicator {
      display: none;
      align-items: center;
      gap: 4px;
      padding: 1px 7px;
      border-radius: 999px;
      background: var(--vscode-badge-background);
      color: var(--vscode-badge-foreground);
      font-size: 11px;
      white-space: nowrap;
    }
    .sel-indicator.active { display: inline-flex; }

    /* Conversation bar */
    .convo-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-bottom: 1px solid var(--border);
    }
    .convo-title { font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .convo-actions { margin-left: auto; display: flex; gap: 6px; }

    /* Index panel */
    .index-panel { border-bottom: 1px solid var(--border); padding: 6px 12px; }
    .index-panel > summary { cursor: pointer; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    .index-body { margin-top: 6px; line-height: 1.5; font-size: 12px; color: var(--muted); white-space: pre-wrap; overflow-wrap: anywhere; }
    .index-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }

    /* Messages */
    .messages { flex: 1; overflow-y: auto; padding: 12px; min-height: 0; }
    .empty-state { color: var(--muted); padding: 24px 12px; text-align: center; line-height: 1.6; }

    .msg { margin-bottom: 16px; }
    .msg-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .avatar {
      width: 22px; height: 22px; border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      flex: 0 0 auto;
    }
    .avatar.user { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
    .avatar.assistant { background: var(--accent); color: var(--vscode-button-foreground, #fff); }
    .avatar.error { background: var(--error); color: var(--vscode-button-foreground, #fff); }
    .msg-author { font-weight: 600; font-size: 12px; }
    .task-chip {
      font-size: 10px; padding: 1px 6px; border-radius: 999px;
      border: 1px solid var(--border); color: var(--muted); text-transform: lowercase;
    }
    .msg-time { margin-left: auto; color: var(--muted); font-size: 11px; }

    .bubble {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px 12px;
      background: var(--vscode-editor-background);
      line-height: 1.55;
      overflow-wrap: anywhere;
    }
    .msg.user .bubble { background: var(--vscode-input-background); }
    .msg.error .bubble { border-color: var(--error); }
    .bubble.plain { white-space: pre-wrap; }

    /* Markdown */
    .md h1, .md h2, .md h3, .md h4 { margin: 12px 0 6px; line-height: 1.3; }
    .md h1 { font-size: 1.25em; } .md h2 { font-size: 1.15em; } .md h3 { font-size: 1.05em; } .md h4 { font-size: 1em; }
    .md p { margin: 6px 0; }
    .md ul, .md ol { margin: 6px 0; padding-left: 20px; }
    .md li { margin: 2px 0; }
    .md a { color: var(--accent); }
    .md code.inline {
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 0.92em;
      background: var(--vscode-textCodeBlock-background);
      padding: 1px 5px; border-radius: 4px;
    }
    .md table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 0.95em; }
    .md th, .md td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; }
    .md th { background: var(--vscode-editorWidget-background, rgba(128,128,128,0.1)); }

    /* Code block */
    .code-block { margin: 10px 0; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
    .code-header {
      display: flex; align-items: center; gap: 6px;
      padding: 5px 8px;
      background: var(--vscode-editorWidget-background, rgba(128,128,128,0.08));
      border-bottom: 1px solid var(--border);
    }
    .code-lang { font-size: 11px; color: var(--muted); text-transform: none; }
    .code-actions { margin-left: auto; display: flex; gap: 4px; }
    pre { margin: 0; padding: 10px; overflow: auto; }
    pre code {
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: var(--vscode-editor-font-size, 0.92em);
      white-space: pre;
      background: var(--vscode-textCodeBlock-background);
      display: block;
    }
    .tok-keyword { color: var(--tok-keyword); }
    .tok-string { color: var(--tok-string); }
    .tok-number { color: var(--tok-number); }
    .tok-comment { color: var(--tok-comment); font-style: italic; }
    .tok-function { color: var(--tok-function); }
    .diff-add { color: var(--vscode-testing-iconPassed, #3fb950); }
    .diff-del { color: var(--error); }
    .diff-meta { color: var(--muted); }

    /* Collapsible sections */
    details.section { margin-top: 8px; border-top: 1px solid var(--border); padding-top: 6px; }
    details.section > summary {
      cursor: pointer; color: var(--muted); font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; list-style: none;
    }
    details.section > summary::-webkit-details-marker { display: none; }
    details.section > summary::before { content: '▸ '; }
    details.section[open] > summary::before { content: '▾ '; }
    .section-body { margin-top: 6px; }
    .summary-box { color: var(--vscode-foreground); }

    .source-item { margin-top: 6px; }
    .source-link {
      color: var(--accent); cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; gap: 5px; overflow-wrap: anywhere;
    }
    .source-link:hover { text-decoration: underline; }
    .source-meta { color: var(--muted); font-size: 11px; margin-top: 1px; }

    /* Context used */
    .context-panel { margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px; }
    .context-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 11px; padding: 2px 8px; border-radius: 999px;
      border: 1px solid var(--border); color: var(--muted);
    }
    .chip.on { color: var(--vscode-testing-iconPassed, #3fb950); border-color: currentColor; }
    .chip.off { opacity: 0.8; }
    .advanced { margin-top: 6px; }
    .advanced > summary { cursor: pointer; color: var(--muted); font-size: 11px; }
    .advanced pre { font-size: 11px; }

    /* Actions */
    .actions-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }

    /* Buttons */
    button {
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 6px;
      padding: 4px 9px;
      color: var(--vscode-button-foreground);
      background: var(--vscode-button-background);
      cursor: pointer;
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button.ghost {
      color: var(--vscode-foreground);
      background: transparent;
      border-color: var(--border);
    }
    button.ghost:hover { background: var(--vscode-toolbar-hoverBackground, rgba(128,128,128,0.15)); }
    button.icon-only { padding: 4px; }
    button:disabled { opacity: 0.55; cursor: default; }
    button svg { width: 14px; height: 14px; }

    /* Composer */
    .composer { border-top: 1px solid var(--border); padding: 8px; background: var(--vscode-sideBar-background); }
    textarea {
      width: 100%; min-height: 64px; max-height: 200px; box-sizing: border-box; resize: vertical;
      color: var(--vscode-input-foreground); background: var(--vscode-input-background);
      border: 1px solid var(--vscode-input-border, var(--border)); border-radius: 6px; padding: 8px;
      font-family: var(--vscode-font-family); font-size: var(--vscode-font-size);
    }
    .composer-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    .hint { color: var(--muted); font-size: 11px; margin-left: auto; }

    /* Spinner */
    .spinner {
      width: 14px; height: 14px; border-radius: 50%;
      border: 2px solid var(--border); border-top-color: var(--accent);
      animation: spin 0.8s linear infinite; display: inline-block;
    }
    .typing { color: var(--muted); display: inline-flex; align-items: center; gap: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* Toast */
    .toast {
      position: fixed; left: 50%; bottom: 84px; transform: translateX(-50%) translateY(10px);
      background: var(--vscode-notifications-background, var(--vscode-editorWidget-background));
      color: var(--vscode-notifications-foreground, var(--vscode-foreground));
      border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 12px;
      opacity: 0; pointer-events: none; transition: opacity 0.2s, transform 0.2s; z-index: 10;
    }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="header-top">
        <div class="project" title="Workspace">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
          <span id="projectName">Workspace</span>
        </div>
        <span id="indexBadge" class="badge"><span class="dot"></span><span id="indexBadgeText">Checking…</span></span>
      </div>
      <div class="header-file">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>
        <span id="activeFile" class="file-name">No active file</span>
        <span id="selIndicator" class="sel-indicator"></span>
      </div>
    </div>

    <div class="convo-bar">
      <span id="convoTitle" class="convo-title">New conversation</span>
      <div class="convo-actions">
        <button id="generateReportBtn" class="ghost" title="Generate project report"><svg viewBox="0 0 24 24" aria-hidden="true" width="14" height="14" style="pointer-events: none;"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M14 3v5h5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9 13h6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9 17h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>Report</button>
        <button id="newConvoBtn" class="ghost icon-only" title="New conversation"><svg viewBox="0 0 24 24" aria-hidden="true" width="14" height="14" style="pointer-events: none;"><path d="M12 5v14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M5 12h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button>
        <button id="clearChatBtn" class="ghost icon-only" title="Clear chat"><svg viewBox="0 0 24 24" aria-hidden="true" width="14" height="14" style="pointer-events: none;"><path d="M5 7h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9 7V5h6v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M7 7l1 13h8l1-13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button>
      </div>
    </div>

    <details id="indexPanel" class="index-panel">
      <summary>Workspace index</summary>
      <div id="indexBody" class="index-body">Checking workspace index…</div>
      <div class="index-actions">
        <button id="checkRagStatus" class="ghost">Check Status</button>
        <button id="indexProject" class="ghost">Index Project</button>
        <button id="fullIndexProject" class="ghost">Full Re-index</button>
        <button id="resetProjectIndex" class="ghost">Reset Index</button>
      </div>
    </details>

    <div id="messages" class="messages" aria-live="polite">
      <div class="empty-state">Ask about the current selection, active file, or indexed workspace.<br/>Press Enter to send, Shift+Enter for a new line.</div>
    </div>

    <div class="composer">
      <textarea id="input" placeholder="Ask me to explain code, fix a bug, or generate a new feature…"></textarea>
      <div class="composer-row">
        <button id="send" class="primary" title="Send (Enter)">Send</button>
        
        <span class="hint" style="margin-left: auto;">Enter to send · Shift+Enter for newline</span>
      </div>
    </div>
  </div>
  <div id="toast" class="toast" role="status"></div>

  <script nonce="${nonce}">
${WEBVIEW_SCRIPT}
  </script>
</body>
</html>`;
}

const WEBVIEW_SCRIPT = String.raw`
(function () {
  var vscode = acquireVsCodeApi();

  window.onerror = function (message, source, lineno, colno, error) {
    try {
      vscode.postMessage({
        type: 'webviewError',
        message: message + ' (' + source + ':' + lineno + ':' + colno + ')'
      });
    } catch (e) {}
  };

  var messages = document.getElementById('messages');
  var input = document.getElementById('input');
  var sendBtn = document.getElementById('send');
    var convoTitle = document.getElementById('convoTitle');
  var indexBadge = document.getElementById('indexBadge');
  var indexBadgeText = document.getElementById('indexBadgeText');
  var indexBody = document.getElementById('indexBody');
  var projectName = document.getElementById('projectName');
  var activeFileEl = document.getElementById('activeFile');
  var selIndicator = document.getElementById('selIndicator');
  var toast = document.getElementById('toast');

  var lastUserInstruction = '';
  var titleSet = false;
  var pendingEl = null;
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var BT = String.fromCharCode(96);
  var FENCE = BT + BT + BT;
  var FENCE_OPEN = new RegExp('^\\s*' + FENCE + '(\\w*)\\s*\$');
  var FENCE_CLOSE = new RegExp('^\\s*' + FENCE + '\\s*\$');
  var FENCE_ANY = new RegExp('^\\s*' + FENCE);
  var INLINE_CODE = new RegExp('(' + BT + '[^' + BT + ']+' + BT + ')');

  var ICONS = {
    user: { paths: ['M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z', 'M5 20a7 7 0 0 1 14 0'] },
    assistant: { filled: true, paths: ['M12 2l2.1 6L20 10l-5.9 2L12 18l-2.1-6L4 10l5.9-2z'] },
    copy: { paths: ['M9 9h11v11H9z', 'M5 15V4h11'] },
    check: { paths: ['M5 13l4 4L19 7'] },
    apply: { paths: ['M12 3v11', 'M8 11l4 4 4-4', 'M5 21h14'] },
    diff: { paths: ['M6 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M18 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M6 9v6a3 3 0 0 0 3 3h6', 'M18 15V9a3 3 0 0 0-3-3H9'] },
    retry: { paths: ['M4 4v6h6', 'M20 9a8 8 0 0 0-15-2', 'M20 20v-6h-6', 'M4 15a8 8 0 0 0 15 2'] },
    explain: { paths: ['M21 12a8 8 0 0 1-8 8H7l-4 3V12a8 8 0 0 1 8-8h2a8 8 0 0 1 8 8z'] },
    sources: { paths: ['M4 6h16', 'M4 12h16', 'M4 18h10'] },
    file: { paths: ['M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z', 'M14 3v5h5'] },
    error: { paths: ['M12 9v4', 'M12 17h.01', 'M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'] },
    plus: { paths: ['M12 5v14', 'M5 12h14'] },
    trash: { paths: ['M5 7h14', 'M9 7V5h6v2', 'M7 7l1 13h8l1-13'] },
    report: { paths: ['M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z', 'M14 3v5h5', 'M9 13h6', 'M9 17h4'] },
    edit: { paths: ['M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7', 'M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z'] },
    stop: { paths: ['M4 4h16v16H4z'] },
    open: { paths: ['M14 4h6v6', 'M20 4l-9 9', 'M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5'] }
  };

  function makeIcon(name) {
    var spec = ICONS[name] || ICONS.file;
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    for (var i = 0; i < spec.paths.length; i++) {
      var p = document.createElementNS(SVG_NS, 'path');
      p.setAttribute('d', spec.paths[i]);
      if (spec.filled) {
        p.setAttribute('fill', 'currentColor');
      } else {
        p.setAttribute('fill', 'none');
        p.setAttribute('stroke', 'currentColor');
        p.setAttribute('stroke-width', '2');
        p.setAttribute('stroke-linecap', 'round');
        p.setAttribute('stroke-linejoin', 'round');
      }
      svg.appendChild(p);
    }
    return svg;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  function button(label, iconName, className) {
    var b = el('button', className || 'ghost');
    if (iconName) { b.appendChild(makeIcon(iconName)); }
    if (label) { b.appendChild(el('span', null, label)); }
    return b;
  }

  function avatar(kind) {
    var a = el('span', 'avatar ' + kind);
    a.appendChild(makeIcon(kind === 'user' ? 'user' : kind === 'error' ? 'error' : 'assistant'));
    return a;
  }

  function nowTime() {
    try { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
    catch (e) { return ''; }
  }

  function clearEmptyState() {
    var empty = messages.querySelector('.empty-state');
    if (empty) { empty.remove(); }
  }

  var followMode = true;
  var jumpBtn = document.getElementById('jumpBtn');
  messages.addEventListener('scroll', function() {
    var dist = messages.scrollHeight - messages.scrollTop - messages.clientHeight;
    if (dist > 80) {
      followMode = false;
      if (jumpBtn) jumpBtn.style.display = 'block';
    } else {
      followMode = true;
      if (jumpBtn) jumpBtn.style.display = 'none';
    }
  });
  if (jumpBtn) {
    jumpBtn.addEventListener('click', function() {
      followMode = true;
      jumpBtn.style.display = 'none';
      messages.scrollTop = messages.scrollHeight;
    });
  }

  function scrollToBottom() { 
    if (followMode) {
      messages.scrollTop = messages.scrollHeight; 
    }
  }

  function showToast(text) {
    toast.textContent = text;
    toast.classList.add('show');
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(function () { toast.classList.remove('show'); }, 1800);
  }

  function displayLang(lang) {
    if (!lang) { return 'Code'; }
    var map = { python: 'Python', javascript: 'JavaScript', typescript: 'TypeScript', java: 'Java', cpp: 'C++', csharp: 'C#', rust: 'Rust', json: 'JSON', yaml: 'YAML', markdown: 'Markdown', text: 'Text', diff: 'Diff' };
    if (map[lang]) { return map[lang]; }
    return lang.charAt(0).toUpperCase() + lang.slice(1);
  }

  // ---- Syntax highlighting (lightweight, dependency-free) ----
  var KEYWORDS = {
    python: 'def class return if elif else for while import from as with try except finally raise yield lambda None True False and or not in is pass break continue global nonlocal assert async await del print self',
    javascript: 'function const let var return if else for while import from export default class extends new try catch finally throw typeof instanceof in of await async yield this super null undefined true false switch case break continue do delete void',
    typescript: 'function const let var return if else for while import from export default class extends new try catch finally throw typeof instanceof in of await async yield this super null undefined true false switch case break continue interface type enum implements public private protected readonly',
    java: 'public private protected class interface enum void int double float boolean char long short return if else for while new try catch finally throw throws import package static final abstract this super null true false switch case break continue',
    cpp: 'class struct enum void int double float bool char long short return if else for while new delete try catch throw namespace using template typename const static public private protected this nullptr true false switch case break continue auto',
    csharp: 'public private protected internal class interface enum void int double float bool char string return if else for while new try catch finally throw using namespace static readonly this base null true false switch case break continue var async await',
    rust: 'fn let mut struct enum impl trait pub use mod match if else for while loop return self crate as where async await move ref dyn const static true false Some None Ok Err'
  };
  var HASH_COMMENT_LANGS = { python: 1, ruby: 1, yaml: 1, toml: 1, shell: 1, bash: 1, dockerfile: 1, config: 1 };

  function highlightInto(codeEl, code, lang) {
    var keywords = {};
    (KEYWORDS[lang] || '').split(' ').forEach(function (k) { if (k) { keywords[k] = 1; } });
    var allowHash = !!HASH_COMMENT_LANGS[lang];
    var token = new RegExp(
      '(\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/' + (allowHash ? '|#[^\\n]*' : '') + ')'
      + '|(' + '"(?:\\\\.|[^"\\\\])*"' + '|' + "'(?:\\\\.|[^'\\\\])*'" + '|' + BT + '(?:\\\\.|[^' + BT + '\\\\])*' + BT + ')'
      + '|(\\b\\d[\\w.]*\\b)'
      + '|([A-Za-z_\$][\\w\$]*)'
      + '|([\\s\\S])',
      'g'
    );
    var m;
    while ((m = token.exec(code)) !== null) {
      if (m[1]) { codeEl.appendChild(el('span', 'tok-comment', m[1])); }
      else if (m[2]) { codeEl.appendChild(el('span', 'tok-string', m[2])); }
      else if (m[3]) { codeEl.appendChild(el('span', 'tok-number', m[3])); }
      else if (m[4]) {
        if (keywords[m[4]]) { codeEl.appendChild(el('span', 'tok-keyword', m[4])); }
        else if (code.charAt(token.lastIndex) === '(') { codeEl.appendChild(el('span', 'tok-function', m[4])); }
        else { codeEl.appendChild(document.createTextNode(m[4])); }
      } else { codeEl.appendChild(document.createTextNode(m[5])); }
    }
  }

  function highlightDiff(codeEl, code) {
    code.split('\n').forEach(function (line, idx) {
      var cls = '';
      if (line.charAt(0) === '+') { cls = 'diff-add'; }
      else if (line.charAt(0) === '-') { cls = 'diff-del'; }
      else if (line.charAt(0) === '@' || line.indexOf('diff ') === 0 || line.indexOf('---') === 0 || line.indexOf('+++') === 0) { cls = 'diff-meta'; }
      codeEl.appendChild(el('span', cls, (idx ? '\n' : '') + line));
    });
  }

  function createCodeBlock(code, lang, opts) {
    opts = opts || {};
    var block = el('div', 'code-block');
    var header = el('div', 'code-header');
    header.appendChild(el('span', 'code-lang', displayLang(lang)));
    var actions = el('div', 'code-actions');

    var copyBtn = button('Copy', 'copy', 'ghost');
    copyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'copyText', text: code }); });
    actions.appendChild(copyBtn);

    if (opts.primary && opts.responseId) {
      var applyBtn = button('Apply', 'apply', 'ghost');
      applyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'apply', responseId: opts.responseId }); });
      actions.appendChild(applyBtn);
      if (opts.hasDiff) {
        var diffBtn = button('Diff', 'diff', 'ghost');
        diffBtn.addEventListener('click', function () { vscode.postMessage({ type: 'openDiff', responseId: opts.responseId }); });
        actions.appendChild(diffBtn);
      }
    }

    header.appendChild(actions);
    block.appendChild(header);

    var pre = el('pre');
    var codeEl = el('code');
    if (lang === 'diff') { highlightDiff(codeEl, code); }
    else { highlightInto(codeEl, code, lang); }
    pre.appendChild(codeEl);
    block.appendChild(pre);
    return block;
  }

  // ---- Markdown rendering (DOM-based, safe) ----
  function appendInline(parent, text) {
    var parts = text.split(INLINE_CODE);
    for (var i = 0; i < parts.length; i++) {
      var part = parts[i];
      if (!part) { continue; }
      if (part.charAt(0) === BT && part.charAt(part.length - 1) === BT) {
        parent.appendChild(el('code', 'inline', part.slice(1, -1)));
      } else {
        appendEmphasis(parent, part);
      }
    }
  }

  function appendEmphasis(parent, text) {
    var re = /(\*\*([^*]+)\*\*)|(__([^_]+)__)|(\*([^*]+)\*)|(\[([^\]]+)\]\(([^)]+)\))/g;
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) { parent.appendChild(document.createTextNode(text.slice(last, m.index))); }
      if (m[1]) { parent.appendChild(el('strong', null, m[2])); }
      else if (m[3]) { parent.appendChild(el('strong', null, m[4])); }
      else if (m[5]) { parent.appendChild(el('em', null, m[6])); }
      else if (m[7]) {
        var link = el('a', null, m[8]);
        link.setAttribute('title', m[9]);
        parent.appendChild(link);
      }
      last = re.lastIndex;
    }
    if (last < text.length) { parent.appendChild(document.createTextNode(text.slice(last))); }
  }

  function isTableSeparator(line) { return /^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*\$/.test(line) || /^\s*\|?(\s*:?-+:?\s*)(\|\s*:?-+:?\s*)+\|?\s*\$/.test(line); }

  function splitRow(line) {
    var trimmed = line.trim().replace(/^\|/, '').replace(/\|\$/, '');
    return trimmed.split('|').map(function (c) { return c.trim(); });
  }

  function renderMarkdown(text) {
    var frag = document.createDocumentFragment();
    var lines = (text || '').replace(/\r\n/g, '\n').split('\n');
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];
      var fence = line.match(FENCE_OPEN);
      if (fence) {
        var lang = fence[1] || '';
        var buf = [];
        i++;
        while (i < lines.length && !FENCE_CLOSE.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        frag.appendChild(createCodeBlock(buf.join('\n'), lang, {}));
        continue;
      }
      var heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        var level = Math.min(heading[1].length, 4);
        var h = el('h' + level);
        appendInline(h, heading[2]);
        frag.appendChild(h);
        i++;
        continue;
      }
      if (line.indexOf('|') !== -1 && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        var table = el('table');
        var thead = el('thead');
        var htr = el('tr');
        splitRow(line).forEach(function (cell) { var th = el('th'); appendInline(th, cell); htr.appendChild(th); });
        thead.appendChild(htr);
        table.appendChild(thead);
        var tbody = el('tbody');
        i += 2;
        while (i < lines.length && lines[i].indexOf('|') !== -1 && lines[i].trim()) {
          var tr = el('tr');
          splitRow(lines[i]).forEach(function (cell) { var td = el('td'); appendInline(td, cell); tr.appendChild(td); });
          tbody.appendChild(tr);
          i++;
        }
        table.appendChild(tbody);
        frag.appendChild(table);
        continue;
      }
      if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
        var ordered = /^\s*\d+\.\s+/.test(line);
        var list = el(ordered ? 'ol' : 'ul');
        while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
          var li = el('li');
          appendInline(li, lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, ''));
          list.appendChild(li);
          i++;
        }
        frag.appendChild(list);
        continue;
      }
      if (!line.trim()) { i++; continue; }
      var paraLines = [];
      while (i < lines.length && lines[i].trim() && !FENCE_ANY.test(lines[i]) && !/^(#{1,6})\s+/.test(lines[i]) && !/^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
        paraLines.push(lines[i]);
        i++;
      }
      if (paraLines.length === 0) {
        // Failsafe: if we failed to consume anything, forcefully increment to prevent infinite loop
        i++;
        continue;
      }
      var p = el('p');
      appendInline(p, paraLines.join(' '));
      frag.appendChild(p);
    }
    return frag;
  }

  // ---- Sources / files ----
  function sourceRelative(source) {
    var md = source && source.metadata && typeof source.metadata === 'object' ? source.metadata : {};
    return String(md.relative_file_path || md.relative_path || source.file_path || 'unknown');
  }

  function sourceLink(source) {
    var link = el('span', 'source-link');
    link.appendChild(makeIcon('file'));
    link.appendChild(el('span', null, sourceRelative(source)));
    link.addEventListener('click', function () {
      vscode.postMessage({ type: 'openFile', filePath: source.file_path || sourceRelative(source), line: source.start_line || 1 });
    });
    return link;
  }

  function buildSection(title, open) {
    var d = el('details', 'section');
    if (open) { d.setAttribute('open', ''); }
    d.appendChild(el('summary', null, title));
    var body = el('div', 'section-body');
    d.appendChild(body);
    return { details: d, body: body };
  }

  function chip(label, state) {
    var c = el('span', 'chip' + (state ? ' ' + state : ''));
    c.appendChild(el('span', 'dot'));
    c.appendChild(el('span', null, label));
    return c;
  }

  function buildContextPanel(response, showTiming) {
    var md = response.metadata || {};
    var panel = el('div', 'context-panel');
    var chips = el('div', 'context-chips');
    var fileCount = typeof md.source_file_count === 'number'
      ? md.source_file_count
      : (Array.isArray(response.rag_sources) ? new Set(response.rag_sources.map(sourceRelative)).size : 0);
    chips.appendChild(chip(fileCount + ' file' + (fileCount === 1 ? '' : 's') + ' retrieved', fileCount ? 'on' : 'off'));
    chips.appendChild(chip(response.used_rag ? 'RAG on' : 'RAG off', response.used_rag ? 'on' : 'off'));
    chips.appendChild(chip(md.project_map_used ? 'Project map' : 'No project map', md.project_map_used ? 'on' : 'off'));
    if (typeof md.timing_rag_ms === 'number') { chips.appendChild(chip(md.timing_rag_ms + ' ms retrieval', null)); }
    panel.appendChild(chips);

    var advanced = el('details', 'advanced');
    if (showTiming) { advanced.setAttribute('open', ''); }
    advanced.appendChild(el('summary', null, 'Advanced metadata'));
    var keys = ['model_name', 'timing_total_ms', 'timing_model_ms', 'timing_validation_ms', 'rag_best_score', 'rag_threshold', 'rag_skip_reason', 'prompt_length_chars', 'generated_length_chars', 'validator_used'];
    var picked = {};
    Object.keys(md).forEach(function (k) { if (md[k] !== undefined && md[k] !== null) { picked[k] = md[k]; } });
    var pre = el('pre');
    pre.appendChild(el('code', null, JSON.stringify(picked, null, 2)));
    advanced.appendChild(pre);
    panel.appendChild(advanced);
    return panel;
  }

  function firstParagraph(text) {
    var blocks = (text || '').replace(/\r\n/g, '\n').split(/\n\s*\n/);
    return { head: (blocks[0] || '').trim(), rest: blocks.slice(1).join('\n\n').trim() };
  }

  // ---- Message builders ----
  function messageRow(kind, author) {
    var row = el('div', 'msg ' + kind);
    var head = el('div', 'msg-head');
    head.appendChild(avatar(kind === 'user' ? 'user' : kind === 'error' ? 'error' : 'assistant'));
    head.appendChild(el('span', 'msg-author', author));
    row._head = head;
    row.appendChild(head);
    return row;
  }

  function appendUser(text) {
    clearEmptyState();
    var row = messageRow('user', 'You');
    row._head.appendChild(el('span', 'msg-time', nowTime()));
    
    var editBtn = button('Edit', 'edit', 'ghost edit-btn');
    editBtn.style.display = 'none'; // hidden by default, handle in css later
    row._head.appendChild(editBtn);

    var bubble = el('div', 'bubble plain');
    var textNode = document.createTextNode(text);
    bubble.appendChild(textNode);
    row.appendChild(bubble);
    messages.appendChild(row);
    
    editBtn.addEventListener('click', function() {
      bubble.style.display = 'none';
      var editorDiv = el('div', 'inline-editor');
      var ta = el('textarea', 'edit-textarea');
      ta.value = text;
      editorDiv.appendChild(ta);
      
      var rowActions = el('div', 'editor-actions');
      var saveBtn = button('Save & Regenerate', null, 'primary');
      var cancelBtn = button('Cancel', null, 'ghost');
      rowActions.appendChild(cancelBtn);
      rowActions.appendChild(saveBtn);
      editorDiv.appendChild(rowActions);
      
      row.appendChild(editorDiv);
      ta.focus();
      
      cancelBtn.addEventListener('click', function() {
        row.removeChild(editorDiv);
        bubble.style.display = 'block';
      });
      
      saveBtn.addEventListener('click', function() {
        var newText = ta.value.trim();
        if (!newText) return;
        text = newText;
        bubble.textContent = newText;
        var editedBadge = el('span', 'edited-badge', ' (Edited)');
        editedBadge.style.fontSize = '0.8em';
        editedBadge.style.opacity = '0.7';
        bubble.appendChild(editedBadge);
        
        row.removeChild(editorDiv);
        bubble.style.display = 'block';
        
        var streamId = row.dataset.streamId;
        if (streamId) {
          vscode.postMessage({ type: 'regenerate', responseId: streamId, text: newText });
        }
      });
    });

    // Make edit button visible on hover
    row.addEventListener('mouseenter', function() { editBtn.style.display = 'inline-flex'; });
    row.addEventListener('mouseleave', function() { editBtn.style.display = 'none'; });

    scrollToBottom();
  }

  function showPending(text) {
    removePending();
    var row = messageRow('assistant', 'Assistant');
    var bubble = el('div', 'bubble');
    var typing = el('div', 'typing');
    typing.appendChild(el('span', 'spinner'));
    typing.appendChild(el('span', null, text || 'Thinking locally…'));
    bubble.appendChild(typing);
    row.appendChild(bubble);
    messages.appendChild(row);
    pendingEl = row;
    scrollToBottom();
  }

  function removePending() {
    if (pendingEl && pendingEl.parentNode) { pendingEl.parentNode.removeChild(pendingEl); }
    pendingEl = null;
  }

  function appendAssistant(payload) {
    removePending();
    clearEmptyState();
    var response = payload.response;
    var existing = messages.querySelector('.msg.assistant[data-stream-id="' + payload.responseId + '"]');
    var row, bubble;
    if (existing) {
      row = existing;
      bubble = row.querySelector('.bubble');
      while (bubble.firstChild) bubble.removeChild(bubble.firstChild);
      while (row._head.firstChild) row._head.removeChild(row._head.firstChild);
      row._head.appendChild(avatar('assistant'));
      row._head.appendChild(el('span', 'msg-author', 'Assistant'));
    } else {
      row = messageRow('assistant', 'Assistant');
      row.dataset.streamId = payload.responseId;
      bubble = el('div', 'bubble');
    }
    
    if (response.task) { row._head.appendChild(el('span', 'task-chip', response.task)); }
    row._head.appendChild(el('span', 'msg-time', nowTime()));
    
    var hasCode = Boolean(response.generated_code && response.generated_code.trim());
    var hasExplanation = Boolean(response.explanation && response.explanation.trim());

    if (hasExplanation) {
      var parts = firstParagraph(response.explanation);
      if (parts.rest) {
        var summary = buildSection('Summary', true);
        var summaryBox = el('div', 'summary-box md');
        summaryBox.appendChild(renderMarkdown(parts.head));
        summary.body.appendChild(summaryBox);
        bubble.appendChild(summary.details);

        var detailed = buildSection('Detailed explanation', true);
        var detailedBox = el('div', 'md');
        detailedBox.appendChild(renderMarkdown(parts.rest));
        detailed.body.appendChild(detailedBox);
        bubble.appendChild(detailed.details);
      } else {
        var only = el('div', 'md');
        only.appendChild(renderMarkdown(response.explanation));
        bubble.appendChild(only);
      }
    } else if (!hasCode) {
      bubble.appendChild(el('div', null, 'No response text returned.'));
    }

    if (response.edits && response.edits.length > 0) {
      var applyAllTopBtn = button('Apply All Edits (' + response.edits.length + ' files)', 'apply', '');
      applyAllTopBtn.addEventListener('click', function () { vscode.postMessage({ type: 'applyAll', responseId: payload.responseId }); });
      applyAllTopBtn.style.marginBottom = '1em';
      applyAllTopBtn.style.width = '100%';
      bubble.appendChild(applyAllTopBtn);

      response.edits.forEach(function (edit) {
        var header = el('div', 'edit-header');
        header.style.fontWeight = 'bold';
        header.style.marginTop = '1em';
        header.style.padding = '5px';
        header.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        header.textContent = edit.file_path;
        bubble.appendChild(header);

        if (edit.reason) {
          var expl = el('div', 'markdown-body');
          expl.style.padding = '0 5px';
          expl.appendChild(renderMarkdown('_Reason: ' + edit.reason + '_'));
          bubble.appendChild(expl);
        }

        bubble.appendChild(createCodeBlock(edit.new_content, response.language || 'text', { primary: false }));
      });
    } else if (hasCode) {
      bubble.appendChild(createCodeBlock(response.generated_code, response.language, { primary: true, responseId: payload.responseId, hasDiff: !!response.diff }));
    }

    if (response.diff) {
      var diffSection = buildSection('Diff preview', false);
      diffSection.body.appendChild(createCodeBlock(response.diff, 'diff', {}));
      bubble.appendChild(diffSection.details);
    }

    var sources = Array.isArray(response.rag_sources) ? response.rag_sources : [];
    var sourcesSection = null;
    if (sources.length) {
      sourcesSection = buildSection('Sources used (' + sources.length + ')', false);
      sources.forEach(function (source) {
        var item = el('div', 'source-item');
        item.appendChild(sourceLink(source));
        var meta = [];
        if (source.symbol_name) { meta.push('symbol ' + source.symbol_name); }
        if (source.start_line || source.end_line) { meta.push('lines ' + (source.start_line || '?') + '-' + (source.end_line || '?')); }
        if (typeof source.score === 'number') { meta.push('score ' + source.score.toFixed(3)); }
        if (meta.length) { item.appendChild(el('div', 'source-meta', meta.join(' · '))); }
        sourcesSection.body.appendChild(item);
      });
      bubble.appendChild(sourcesSection.details);

      var unique = [];
      var seen = {};
      sources.forEach(function (source) {
        var rel = sourceRelative(source);
        if (!seen[rel]) { seen[rel] = 1; unique.push(source); }
      });
      var filesSection = buildSection('Retrieved files (' + unique.length + ')', false);
      unique.forEach(function (source) {
        var item = el('div', 'source-item');
        item.appendChild(sourceLink(source));
        filesSection.body.appendChild(item);
      });
      bubble.appendChild(filesSection.details);
    }

    var validation = response.validation;
    if (validation && validation.tests_executed) {
      var testPanel = el('div', 'context-panel');
      var testChips = el('div', 'context-chips');
      var framework = validation.test_framework || 'tests';
      var passed = validation.tests_passed === true;
      var failed = validation.tests_passed === false;
      testChips.appendChild(chip(framework + ' tests', passed ? 'on' : failed ? 'off' : null));
      testChips.appendChild(chip((validation.tests_run || 0) + ' run', null));
      testChips.appendChild(chip((validation.tests_failed || 0) + ' failed', (validation.tests_failed || 0) ? 'off' : 'on'));
      if (typeof validation.test_duration_ms === 'number') {
        testChips.appendChild(chip((validation.test_duration_ms / 1000).toFixed(1) + 's', null));
      }
      testPanel.appendChild(testChips);
      bubble.appendChild(testPanel);
    }

    if (response.metadata) {
      bubble.appendChild(buildContextPanel(response, payload.showTiming));
    }

    // Result actions
    var actions = el('div', 'actions-row');
    var copyBtn = button('Copy', 'copy', 'ghost');
    copyBtn.addEventListener('click', function () {
      var combined = [];
      if (hasExplanation) { combined.push(response.explanation.trim()); }
      if (hasCode) { combined.push(response.generated_code.trim()); }
      vscode.postMessage({ type: 'copyText', text: combined.join('\n\n') });
    });
    actions.appendChild(copyBtn);

    var retryBtn = button('Retry', 'retry', 'ghost');
    retryBtn.addEventListener('click', function () { if (lastUserInstruction) { submitText(lastUserInstruction); } });
    actions.appendChild(retryBtn);

    var explainBtn = button('Explain further', 'explain', 'ghost');
    explainBtn.addEventListener('click', function () { submitText('Explain the previous answer in more detail.'); });
    actions.appendChild(explainBtn);

    if (sourcesSection) {
      var srcBtn = button('Show sources', 'sources', 'ghost');
      srcBtn.addEventListener('click', function () {
        sourcesSection.details.setAttribute('open', '');
        sourcesSection.details.scrollIntoView({ block: 'nearest' });
      });
      actions.appendChild(srcBtn);
    }

    if (response.edits && response.edits.length > 0) {
      var applyAllBtn = button('Apply All Edits', 'apply', '');
      applyAllBtn.addEventListener('click', function () { vscode.postMessage({ type: 'applyAll', responseId: payload.responseId }); });
      actions.appendChild(applyAllBtn);
    } else if (hasCode) {
      var applyBtn = button('Apply changes', 'apply', '');
      applyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'apply', responseId: payload.responseId }); });
      actions.appendChild(applyBtn);
    }

    bubble.appendChild(actions);
    if (!existing) {
      row.appendChild(bubble);
      messages.appendChild(row);
    }
    scrollToBottom();
  }

  function appendError(message) {
    removePending();
    clearEmptyState();
    var row = messageRow('error', 'Error');
    row._head.appendChild(el('span', 'msg-time', nowTime()));
    var bubble = el('div', 'bubble', message);
    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function appendStandaloneSources(sources, message) {
    removePending();
    clearEmptyState();
    var row = messageRow('assistant', 'Sources');
    row._head.appendChild(el('span', 'msg-time', nowTime()));
    var bubble = el('div', 'bubble');
    bubble.appendChild(el('div', null, message || 'Sources used for the previous response.'));
    if (Array.isArray(sources) && sources.length) {
      var section = buildSection('Sources used (' + sources.length + ')', true);
      sources.forEach(function (source) {
        var item = el('div', 'source-item');
        item.appendChild(sourceLink(source));
        section.body.appendChild(item);
      });
      bubble.appendChild(section.details);
    }
    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function appendReport(payload) {
    removePending();
    clearEmptyState();
    if (payload.responseId) {
      var existing = messages.querySelector('.msg.assistant[data-stream-id="' + payload.responseId + '"]');
      if (existing && existing.parentNode) {
        existing.parentNode.removeChild(existing);
      }
    }
    var row = messageRow('assistant', 'Project Report');
    row._head.appendChild(el('span', 'task-chip', 'report'));
    row._head.appendChild(el('span', 'msg-time', nowTime()));
    var bubble = el('div', 'bubble');

    if (payload.summary) {
      var summaryBox = el('div', 'summary-box md');
      summaryBox.appendChild(renderMarkdown(payload.summary));
      bubble.appendChild(summaryBox);
    }

    var chips = el('div', 'context-chips');
    var fileCount = payload.filesAnalyzed || 0;
    chips.appendChild(chip(fileCount + ' file' + (fileCount === 1 ? '' : 's') + ' analyzed', fileCount ? 'on' : 'off'));
    chips.appendChild(chip(payload.ragEnabled ? 'RAG on' : 'RAG off', payload.ragEnabled ? 'on' : 'off'));
    chips.appendChild(chip(payload.projectMapUsed ? 'Project map' : 'No project map', payload.projectMapUsed ? 'on' : 'off'));
    if (typeof payload.durationMs === 'number') {
      chips.appendChild(chip((payload.durationMs / 1000).toFixed(1) + 's generation', null));
    }
    bubble.appendChild(chips);

    if (payload.reportPath) {
      bubble.appendChild(el('div', 'source-meta', 'Saved to ' + payload.reportPath));
    }

    var actions = el('div', 'actions-row');
    var openBtn = button('Open Report', 'open', 'primary');
    openBtn.addEventListener('click', function () { vscode.postMessage({ type: 'openReport' }); });
    actions.appendChild(openBtn);
    var regenBtn = button('Regenerate', 'retry', 'ghost');
    regenBtn.addEventListener('click', function () { vscode.postMessage({ type: 'generateReport' }); });
    actions.appendChild(regenBtn);
    var copyBtn = button('Copy Summary', 'copy', 'ghost');
    copyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'copyText', text: payload.summary || '' }); });
    actions.appendChild(copyBtn);
    bubble.appendChild(actions);

    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  // ---- Status / header ----
  function setIndexBadge(state, text) {
    indexBadge.className = 'badge ' + (state || '');
    indexBadgeText.textContent = text;
  }

  function renderRagStatus(payload) {
    if (payload.state === 'indexing') {
      setIndexBadge('busy', 'Indexing…');
      if (payload.message) { indexBody.textContent = payload.message; }
      return;
    }
    if (payload.error) {
      setIndexBadge('warn', 'Unavailable');
      indexBody.textContent = payload.error;
      return;
    }
    var status = payload.status;
    if (!status) {
      setIndexBadge('warn', 'Not indexed');
      indexBody.textContent = payload.message || 'RAG status unavailable.';
      return;
    }
    var ready = status.indexed && status.project_map_exists;
    setIndexBadge(ready ? 'ok' : 'warn', ready ? 'Indexed' : 'Not indexed');
    var lines = [
      'Points: ' + status.point_count + ' · Project map: ' + (status.project_map_exists ? 'yes' : 'no'),
      'Last indexed: ' + (status.last_indexed || 'unknown')
    ];
    if (status.qdrant_ready === false) { lines.push('Qdrant offline or unavailable'); }
    if (Array.isArray(status.frameworks) && status.frameworks.length) { lines.push('Frameworks: ' + status.frameworks.join(', ')); }
    if (payload.message) { lines.push(payload.message); }
    indexBody.textContent = lines.join('\n');
  }

  function renderEditorState(payload) {
    projectName.textContent = payload.projectName || 'Workspace';
    activeFileEl.textContent = payload.activeFile || 'No active file';
    activeFileEl.title = payload.activeFileRelative || payload.activeFile || '';
    if (payload.hasSelection) {
      selIndicator.classList.add('active');
      selIndicator.textContent = payload.selectionLabel || 'selection';
    } else {
      selIndicator.classList.remove('active');
      selIndicator.textContent = '';
    }
  }

  // ---- Conversation ----
  function setTitleIfFirst(text) {
    if (titleSet) { return; }
    var clean = text.replace(/\s+/g, ' ').trim();
    convoTitle.textContent = clean.length > 60 ? clean.slice(0, 60) + '…' : clean;
    titleSet = true;
  }

  function resetConversation() {
    while (messages.firstChild) { messages.removeChild(messages.firstChild); }
    var empty = el('div', 'empty-state');
    empty.appendChild(document.createTextNode('Ask about the current selection, active file, or indexed workspace.'));
    messages.appendChild(empty);
    convoTitle.textContent = 'New conversation';
    titleSet = false;
    lastUserInstruction = '';
  }

  function setLoading(isLoading) {
    sendBtn.style.display = isLoading ? 'none' : 'inline-flex';
        input.disabled = isLoading;
  }

  function submitText(text) {
    var clean = (text || '').trim();
    if (!clean) { return; }
    appendUser(clean);
    setTitleIfFirst(clean);
    lastUserInstruction = clean;
    setLoading(true);
    showPending();
    vscode.postMessage({ type: 'send', text: clean });
  }

  function submit() {
    var text = input.value.trim();
    if (!text) { 
      showToast('Please enter an instruction before sending.');
      return; 
    }
    input.value = '';
    submitText(text);
  }

  // ---- Wiring ----
  sendBtn.addEventListener('click', submit);
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });

  // Remove old JS icon logic for the header buttons
  var generateReportBtn = document.getElementById('generateReportBtn');
  generateReportBtn.addEventListener('click', function () { vscode.postMessage({ type: 'generateReport' }); });
  document.getElementById('newConvoBtn').addEventListener('click', function () {
    resetConversation();
    vscode.postMessage({ type: 'newConversation' });
  });
  document.getElementById('clearChatBtn').addEventListener('click', function () {
    while (messages.firstChild) { messages.removeChild(messages.firstChild); }
    resetConversation();
  });
  document.getElementById('checkRagStatus').addEventListener('click', function () { vscode.postMessage({ type: 'checkRagStatus' }); });
  document.getElementById('indexProject').addEventListener('click', function () { vscode.postMessage({ type: 'indexProject', mode: 'incremental' }); });
  document.getElementById('fullIndexProject').addEventListener('click', function () { vscode.postMessage({ type: 'indexProject', mode: 'full' }); });
  document.getElementById('resetProjectIndex').addEventListener('click', function () { vscode.postMessage({ type: 'resetProjectIndex' }); });

  var streamStates = {};

  function handleStreamStart(payload) {
    if (document.querySelector('.msg.pending')) {
      var p = document.querySelector('.msg.pending');
      if (p && p.parentNode) p.parentNode.removeChild(p);
    }
    
    // Tag the last user message if not already tagged
    var userRows = messages.querySelectorAll('.msg.user');
    if (userRows.length > 0) {
      var lastUser = userRows[userRows.length - 1];
      if (!lastUser.dataset.streamId) {
        lastUser.dataset.streamId = payload.responseId;
      }
    }

    // If targeted regeneration, replace the existing assistant bubble
    var existingAssistant = messages.querySelector('.msg.assistant[data-stream-id="' + payload.responseId + '"]');
    var row, bubble, content_div;

    if (existingAssistant) {
      row = existingAssistant;
      bubble = row.querySelector('.bubble');
      while (bubble.firstChild) bubble.removeChild(bubble.firstChild);
    } else {
      row = messageRow('assistant', 'AI Code Assistant');
      row.dataset.streamId = payload.responseId;
      bubble = el('div', 'bubble assistant');
    }
    
    var head = el('div', 'bubble-head');
    var badge = el('div', 'badge busy');
    badge.appendChild(el('div', 'dot'));
    var badgeText = document.createTextNode('Thinking locally...');
    badge.appendChild(badgeText);
    head.appendChild(badge);
    
    bubble.appendChild(head);
    
    content_div = el('div', 'content markdown-body');
    bubble.appendChild(content_div);
    if (!existingAssistant) {
      row.appendChild(bubble);
      messages.appendChild(row);
    }
    scrollToBottom();
    
    streamStates[payload.responseId] = { text: '', el: content_div, lastRender: 0 };
  }

  function handleStreamToken(payload) {
    var state = streamStates[payload.responseId];
    if (!state || state.stopped) return;
    state.text += payload.content;
    
    var now = Date.now();
    if (now - state.lastRender > 250) {
      while (state.el.firstChild) { state.el.removeChild(state.el.firstChild); }
      state.el.appendChild(renderMarkdown(state.text));
      state.lastRender = now;
      scrollToBottom();
    }
  }

  
  window.addEventListener('message', function (event) {
    var message = event.data;
    if (message.type === 'stream_start') { handleStreamStart(message.payload); }
    else if (message.type === 'stream_token') { handleStreamToken(message.payload); }
    else if (message.type === 'assistant') { 
      var existing = messages.querySelector('.msg.assistant[data-stream-id="' + message.payload.responseId + '"]');
      if (existing && existing.parentNode) {
         existing.parentNode.removeChild(existing);
      }
      setLoading(false); 
      appendAssistant(message.payload); 
    }
    else if (message.type === 'sources') { setLoading(false); appendStandaloneSources(message.sources, message.message); }
    else if (message.type === 'ragStatus') { renderRagStatus(message.payload); }
    else if (message.type === 'ragIndexing') { setIndexBadge('busy', 'Indexing…'); indexBody.textContent = message.message; }
    else if (message.type === 'editorState') { renderEditorState(message.payload); }
    else if (message.type === 'reportGenerating') { setLoading(true); showPending('Generating project report…'); }
    else if (message.type === 'report') { setLoading(false); appendReport(message.payload); }
    else if (message.type === 'error') { setLoading(false); appendError(message.message); }
    else if (message.type === 'applied') { showToast(message.message); }
    else if (message.type === 'notice') { showToast(message.message); }
    else if (message.type === 'cancelled') {
      setLoading(false);
      showToast(message.message);
      if (pendingEl && pendingEl.parentNode) {
        pendingEl.parentNode.removeChild(pendingEl);
        pendingEl = null;
      }
      var existing = messages.querySelector('.msg.assistant[data-stream-id]');
      if (existing && existing.parentNode) {
         existing.parentNode.removeChild(existing);
      }
      input.value = lastUserInstruction;
      input.focus();
    }
    else if (message.type === 'conversationReset') { resetConversation(); }
  });

  // Load initial states
  vscode.postMessage({ type: 'checkRagStatus' });
  vscode.postMessage({ type: 'requestEditorState' });
}());
`;

function getNonce() {
  let text = "";
  const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
