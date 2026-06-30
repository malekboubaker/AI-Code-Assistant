import * as path from 'node:path';
import * as vscode from 'vscode';
import { ApiClient, ChatHistoryMessage, GenerateRequest, GenerateResponse, RagIndexMode } from './apiClient';
import { AssistantResponsePayload, ChatViewMessage, getChatViewHtml } from './chatPanel';
import { CollectedEditorContext, collectEditorContext, deserializeRange } from './contextCollector';

const CHAT_VIEW_ID = 'aiCodeAssistant.chatView';
const PREVIEW_SCHEME = 'ai-code-assistant-preview';
const MAX_CHAT_HISTORY_MESSAGES = 10;
const MAX_CHAT_HISTORY_ENTRY_CHARS = 1400;

interface PendingApply {
  context: CollectedEditorContext;
  response: GenerateResponse;
}

export function registerChatView(context: vscode.ExtensionContext): void {
  const provider = new ChatViewProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(CHAT_VIEW_ID, provider, {
      webviewOptions: {
        retainContextWhenHidden: true
      }
    }),
    vscode.commands.registerCommand('aiCodeAssistant.openChat', async () => {
      await vscode.commands.executeCommand('workbench.view.extension.aiCodeAssistant');
      await vscode.commands.executeCommand(`${CHAT_VIEW_ID}.focus`);
    })
  );
}

class ChatViewProvider implements vscode.WebviewViewProvider {
  private webviewView?: vscode.WebviewView;
  private responseCounter = 0;
  private lastBackendResponse?: GenerateResponse;
  private chatHistory: ChatHistoryMessage[] = [];
  private currentClient?: ApiClient;
  private currentResponseId?: string;
  private lastReportText?: string;
  private readonly pendingApply = new Map<string, PendingApply>();
  private readonly previewProvider = new GeneratedCodePreviewProvider();

  constructor(private readonly extensionContext: vscode.ExtensionContext) {
    this.extensionContext.subscriptions.push(
      vscode.workspace.registerTextDocumentContentProvider(PREVIEW_SCHEME, this.previewProvider)
    );
  }

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.webviewView = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionContext.extensionUri]
    };
    webviewView.webview.html = getChatViewHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((message: ChatViewMessage) => {
      void this.handleMessage(message);
    });
    
    this.extensionContext.subscriptions.push(
      vscode.window.onDidChangeActiveTextEditor(() => this.sendEditorState()),
      vscode.window.onDidChangeTextEditorSelection(() => this.sendEditorState())
    );

    this.sendEditorState();
    void this.checkRagStatus({ autoIndexIfNeeded: true });
  }

  private sendEditorState(): void {
    if (!this.webviewView) { return; }
    
    const context = collectEditorContext();
    const projectName = getWorkspaceProjectPath() ? path.basename(getWorkspaceProjectPath()!) : 'Workspace';
    
    let activeFile = 'No active file';
    let activeFileRelative = '';
    let hasSelection = false;
    let selectionLabel = '';

    const editor = vscode.window.activeTextEditor;
    if (editor) {
      activeFile = path.basename(editor.document.fileName);
      activeFileRelative = vscode.workspace.asRelativePath(editor.document.fileName);
      hasSelection = !editor.selection.isEmpty;
      if (hasSelection) {
        const start = editor.selection.start.line + 1;
        const end = editor.selection.end.line + 1;
        selectionLabel = start === end ? `:${start}` : `:${start}-${end}`;
        activeFile += selectionLabel;
      }
    }

    this.postMessage({
      type: 'editorState',
      payload: {
        projectName,
        activeFile,
        activeFileRelative,
        hasSelection,
        selectionLabel
      }
    });
  }

  private async handleMessage(message: ChatViewMessage): Promise<void> {
    try {
      if (message.type === 'send') {
        await this.sendChatRequest(message.text);
        return;
      }
      if (message.type === 'stopGeneration') {
        await this.stopGeneration();
        return;
      }
      if (message.type === 'apply') {
        await this.applyGeneratedCode(message.responseId);
        return;
      }
      if (message.type === 'applyAll') {
        await this.applyWorkspaceEdits(message.responseId);
        return;
      }
      if (message.type === 'checkRagStatus') {
        await this.checkRagStatus();
        return;
      }
      if (message.type === 'newConversation') {
        this.chatHistory = [];
        return;
      }
      if (message.type === 'generateReport') {
        await this.generateProjectReport();
        return;
      }
      if (message.type === 'requestEditorState') {
        this.sendEditorState();
        return;
      }
      if (message.type === 'openReport') {
        const doc = await vscode.workspace.openTextDocument({
          content: this.lastReportText || 'No report generated yet.',
          language: 'markdown'
        });
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
        return;
      }
      if (message.type === 'copyText') {
        await vscode.env.clipboard.writeText(message.text);
        vscode.window.showInformationMessage('Copied to clipboard');
        return;
      }
      if (message.type === 'openFile') {
        const workspacePath = getWorkspaceProjectPath();
        if (workspacePath) {
          const fullPath = path.isAbsolute(message.filePath) ? message.filePath : path.join(workspacePath, message.filePath);
          try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(fullPath));
            const editor = await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
            if (message.line) {
              const pos = new vscode.Position(message.line - 1, 0);
              editor.selection = new vscode.Selection(pos, pos);
              editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
            }
          } catch (e) {
            console.error(`Failed to open file: ${fullPath}`, e);
          }
        }
        return;
      }
      if (message.type === 'openDiff') {
        const pending = this.pendingApply.get(message.responseId);
        if (pending && pending.response.diff) {
          const doc = await vscode.workspace.openTextDocument({
            content: pending.response.diff,
            language: 'diff'
          });
          await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
        }
        return;
      }
      if (message.type === 'webviewError') {
        console.error(`AI Code Assistant (Webview Error): ${message.message}`);
        return;
      }
      if (message.type === 'indexProject') {
        await this.indexProject(message.mode);
        return;
      }
      if (message.type === 'resetProjectIndex') {
        await this.resetProjectIndex();
      }
    } catch (error) {
      this.postError(error);
    }
  }

  private async sendChatRequest(instruction: string): Promise<void> {
    if (isSourceListingRequest(instruction)) {
      this.postPreviousSources();
      return;
    }

    const editorContext = collectEditorContext();
    const config = vscode.workspace.getConfiguration('aiCodeAssistant');
    const endpoint = config.get<string>('backendUrl', 'http://localhost:8000/api/v1/generate');
    const timeoutMs = config.get<number>('timeoutMs', 120000);
    const useRag = config.get<boolean>('chat.useRag', true);
    const showTiming = config.get<boolean>('chat.showTimingMetadata', false);

    let client: ApiClient;
    try {
      client = new ApiClient(endpoint, timeoutMs);
    } catch (error) {
      this.postError(formatBackendError(error));
      return;
    }

    const responseId = this.nextResponseId();
    this.currentClient = client;
    this.currentResponseId = responseId;

    const request: GenerateRequest = {
      response_id: responseId,
      instruction,
      code: editorContext.code,
      language: editorContext.language,
      task: null,
      file_path: editorContext.filePath,
      project_path: editorContext.projectPath,
      has_selection: editorContext.hasSelection,
      surrounding_context: editorContext.surroundingContext,
      chat_history: this.chatHistory.slice(),
      use_rag: useRag
    };

    try {
      this.postMessage({ type: 'generatingStarted' });
      const response = await client.streamGenerate(request, (event) => {
        if (event.type === 'stream_start' || event.type === 'stream_token') {
          this.postMessage({ type: event.type, payload: event.payload });
        }
      });
      this.currentClient = undefined;
      this.currentResponseId = undefined;
      this.lastBackendResponse = response;
      if (response.generated_code.trim() || (response.edits && response.edits.length > 0)) {
        this.pendingApply.set(responseId, { context: editorContext, response });
      }
      this.rememberTurn(instruction, editorContext, response);
      const payload: AssistantResponsePayload = {
        responseId,
        instruction,
        response,
        contextSummary: editorContext.contextSummary,
        showTiming
      };
      this.postMessage({ type: 'assistant', payload });

    } catch (error) {
      this.currentClient = undefined;
      this.currentResponseId = undefined;
      this.postError(formatBackendError(error));
    }
  }

  private async stopGeneration(): Promise<void> {
    if (this.currentClient && this.currentResponseId) {
      try {
        await this.currentClient.cancelRequest(this.currentResponseId);
      } catch (error) {
        console.error('Failed to cancel generation:', error);
      }
    }
  }

  private postPreviousSources(): void {
    const sources = this.lastBackendResponse?.rag_sources ?? [];
    this.postMessage({
      type: 'sources',
      sources,
      message: sources.length
        ? 'Sources used for the previous response.'
        : 'No RAG sources are available for the previous response.'
    });
  }

  private async checkRagStatus(options: { autoIndexIfNeeded?: boolean } = {}): Promise<void> {
    const projectPath = getWorkspaceProjectPath();
    if (!projectPath) {
      this.postMessage({
        type: 'ragStatus',
        payload: { message: 'Open a workspace folder to use project RAG indexing.' }
      });
      return;
    }

    let client: ApiClient;
    try {
      client = this.createApiClient();
    } catch (error) {
      this.postMessage({ type: 'ragStatus', payload: { error: formatBackendError(error) } });
      return;
    }

    try {
      const status = await client.getRagStatus(projectPath);
      const ready = status.indexed && status.project_map_exists;
      this.postMessage({
        type: 'ragStatus',
        payload: {
          status,
          message: ready ? 'RAG index ready.' : 'RAG index not ready. Use Index Project to build it.'
        }
      });

      const autoIndex = vscode.workspace
        .getConfiguration('aiCodeAssistant')
        .get<boolean>('rag.autoIndexOnOpen', true);
      if (options.autoIndexIfNeeded && autoIndex && !ready) {
        void this.indexProject('incremental');
      }
    } catch (error) {
      this.postMessage({ type: 'ragStatus', payload: { error: formatBackendError(error) } });
    }
  }

  private async generateProjectReport(): Promise<void> {
    const projectPath = getWorkspaceProjectPath();
    if (!projectPath) {
      this.postError('Cannot generate report: No workspace folder opened.');
      return;
    }

    const config = vscode.workspace.getConfiguration('aiCodeAssistant');
    const endpoint = config.get<string>('backendUrl', 'http://localhost:8000/api/v1/generate');
    const timeoutMs = config.get<number>('timeoutMs', 120000);

    let client: ApiClient;
    try {
      client = new ApiClient(endpoint, timeoutMs);
    } catch (error) {
      this.postError(formatBackendError(error));
      return;
    }

    try {
      this.postMessage({ type: 'reportGenerating' });
      const startTime = Date.now();
      const responseId = this.nextResponseId();
      const request: GenerateRequest = {
        response_id: responseId,
        instruction: 'Generate a comprehensive technical report of this project based on the provided project map. Summarize the architecture, primary languages, frameworks, and key entry points.',
        code: '',
        language: '',
        task: 'project_explain',
        project_path: projectPath,
        use_rag: true
      };

      const response = await client.streamGenerate(request, (event) => {
        if (event.type === 'stream_start' || event.type === 'stream_token') {
          this.postMessage({ type: event.type, payload: event.payload });
        }
      });
      
      this.lastReportText = response.explanation;
      
      this.postMessage({
        type: 'report',
        payload: {
          responseId,
          summary: response.explanation,
          filesAnalyzed: response.rag_sources?.length || 0,
          ragEnabled: true,
          projectMapUsed: true,
          durationMs: Date.now() - startTime
        }
      });
    } catch (error) {
      this.postError(`Failed to generate report: ${formatBackendError(error)}`);
    }
  }

  private async indexProject(mode: RagIndexMode): Promise<void> {
    const projectPath = getWorkspaceProjectPath();
    if (!projectPath) {
      this.postError('Open a workspace folder before indexing a project.');
      return;
    }
    let client: ApiClient;
    try {
      client = this.createApiClient();
    } catch (error) {
      this.postError(error);
      return;
    }

    this.postMessage({
      type: 'ragIndexing',
      message: mode === 'full' ? 'Full project re-indexing started...' : 'Incremental project indexing started...'
    });
    try {
      const response = await client.indexProject(projectPath, mode);
      this.postMessage({
        type: 'ragIndexing',
        message:
          `Indexing complete: ${response.files_indexed}/${response.files_scanned} files indexed, ` +
          `${response.chunks_stored} chunks stored in ${response.duration_ms} ms.`
      });
      await this.checkRagStatus();
    } catch (error) {
      this.postMessage({ type: 'ragStatus', payload: { error: formatBackendError(error) } });
    }
  }

  private async resetProjectIndex(): Promise<void> {
    const projectPath = getWorkspaceProjectPath();
    if (!projectPath) {
      this.postError('Open a workspace folder before resetting a project index.');
      return;
    }
    const choice = await vscode.window.showWarningMessage(
      'Reset the RAG index for this workspace only?',
      { modal: true },
      'Reset Project Index'
    );
    if (choice !== 'Reset Project Index') {
      return;
    }

    let client: ApiClient;
    try {
      client = this.createApiClient();
    } catch (error) {
      this.postError(error);
      return;
    }

    try {
      const response = await client.resetProjectIndex(projectPath);
      this.postMessage({
        type: 'ragIndexing',
        message: `Reset complete for project ${response.project_id}. Deleted points: ${response.deleted_points}.`
      });
      await this.checkRagStatus();
    } catch (error) {
      this.postMessage({ type: 'ragStatus', payload: { error: formatBackendError(error) } });
    }
  }

  private async applyGeneratedCode(responseId: string): Promise<void> {
    const pending = this.pendingApply.get(responseId);
    if (!pending) {
      this.postError('No generated code is available to apply.');
      return;
    }

    const generatedCode = pending.response.generated_code;
    if (!generatedCode.trim()) {
      this.postError('The generated code is empty.');
      return;
    }

    if (pending.context.selection && pending.context.documentUri) {
      await this.replaceSelection(pending.context, generatedCode);
      this.postMessage({ type: 'applied', message: 'Replaced the selected code.' });
      return;
    }

    await this.previewAndApplyFullFile(pending.context, generatedCode);
  }

  private async applyWorkspaceEdits(responseId: string): Promise<void> {
    const pending = this.pendingApply.get(responseId);
    if (!pending || !pending.response || !pending.response.edits) {
      this.postError('No workspace edits available to apply.');
      return;
    }

    const edits = pending.response.edits;
    const workspaceEdit = new vscode.WorkspaceEdit();
    
    // Resolve the active project path dynamically from the context
    const activeProjectPath = pending.context.projectPath;
    const fallbackFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const basePath = activeProjectPath || fallbackFolder;

    if (!basePath) {
      this.postError('No workspace folder open.');
      return;
    }

    for (const edit of edits) {
      const uri = path.isAbsolute(edit.file_path)
        ? vscode.Uri.file(edit.file_path)
        : vscode.Uri.file(path.join(basePath, edit.file_path));

      try {
        const document = await vscode.workspace.openTextDocument(uri);
        const fullRange = new vscode.Range(
          document.positionAt(0),
          document.positionAt(document.getText().length)
        );
        workspaceEdit.replace(uri, fullRange, edit.new_content);
      } catch (err) {
        workspaceEdit.createFile(uri, { ignoreIfExists: true });
        workspaceEdit.insert(uri, new vscode.Position(0, 0), edit.new_content);
      }
    }

    const success = await vscode.workspace.applyEdit(workspaceEdit);
    if (success) {
      this.postMessage({ type: 'applied', message: `Applied workspace edits across ${edits.length} files.` });
    } else {
      this.postError('Failed to apply workspace edits.');
    }
  }

  private async replaceSelection(context: CollectedEditorContext, generatedCode: string): Promise<void> {
    const document = await vscode.workspace.openTextDocument(vscode.Uri.parse(context.documentUri!));
    const editor = await vscode.window.showTextDocument(document, { preview: false });
    await editor.edit((editBuilder) => {
      editBuilder.replace(deserializeRange(context.selection!), generatedCode);
    });
  }

  private async previewAndApplyFullFile(context: CollectedEditorContext, generatedCode: string): Promise<void> {
    let targetUri = context.documentUri ? vscode.Uri.parse(context.documentUri) : undefined;
    if (!targetUri && context.filePath) {
      targetUri = vscode.Uri.file(context.filePath);
    }
    if (!targetUri) {
      const chosen = await vscode.window.showOpenDialog({
        title: 'Choose a file to preview applying the AI Code Assistant output',
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: false
      });
      targetUri = chosen?.[0];
    }
    if (!targetUri) {
      vscode.window.showWarningMessage('AI Code Assistant: select code or choose a file before applying this result.');
      return;
    }

    const document = await vscode.workspace.openTextDocument(targetUri);
    const previewUri = this.previewProvider.setContent(
      generatedCode,
      path.basename(targetUri.fsPath || targetUri.path)
    );
    await vscode.commands.executeCommand(
      'vscode.diff',
      document.uri,
      previewUri,
      `AI Code Assistant Preview: ${path.basename(document.fileName)}`
    );

    const choice = await vscode.window.showWarningMessage(
      'Preview opened. Apply generated code as a full-file replacement? For targeted edits, select code first.',
      { modal: true },
      'Apply to File'
    );
    if (choice !== 'Apply to File') {
      return;
    }

    const fullRange = new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length));
    const edit = new vscode.WorkspaceEdit();
    edit.replace(document.uri, fullRange, generatedCode);
    const applied = await vscode.workspace.applyEdit(edit);
    if (!applied) {
      throw new Error('VS Code rejected the workspace edit.');
    }
    this.postMessage({ type: 'applied', message: 'Applied generated code as a full-file replacement.' });
  }

  private nextResponseId(): string {
    this.responseCounter += 1;
    return `${Date.now()}-${this.responseCounter}`;
  }

  private postError(error: unknown): void {
    this.postMessage({ type: 'error', message: typeof error === 'string' ? error : formatBackendError(error) });
  }

  private postMessage(message: unknown): void {
    void this.webviewView?.webview.postMessage(message);
  }

  private createApiClient(): ApiClient {
    const config = vscode.workspace.getConfiguration('aiCodeAssistant');
    const endpoint = config.get<string>('backendUrl', 'http://localhost:8000/api/v1/generate');
    const timeoutMs = config.get<number>('timeoutMs', 300000);
    return new ApiClient(endpoint, timeoutMs);
  }

  private rememberTurn(
    instruction: string,
    editorContext: CollectedEditorContext,
    response: GenerateResponse
  ): void {
    this.chatHistory.push({
      role: 'user',
      content: trimMemoryEntry(`${instruction}\nContext: ${editorContext.contextSummary}`)
    });
    this.chatHistory.push({
      role: 'assistant',
      content: trimMemoryEntry(formatAssistantMemory(response))
    });
    if (this.chatHistory.length > MAX_CHAT_HISTORY_MESSAGES) {
      this.chatHistory = this.chatHistory.slice(-MAX_CHAT_HISTORY_MESSAGES);
    }
  }
}

class GeneratedCodePreviewProvider implements vscode.TextDocumentContentProvider {
  private readonly contentByUri = new Map<string, string>();
  private readonly onDidChangeEmitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.onDidChangeEmitter.event;

  setContent(content: string, fileName: string): vscode.Uri {
    const uri = vscode.Uri.from({
      scheme: PREVIEW_SCHEME,
      path: `/${fileName}`,
      query: `t=${Date.now()}`
    });
    this.contentByUri.set(uri.toString(), content);
    this.onDidChangeEmitter.fire(uri);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contentByUri.get(uri.toString()) ?? '';
  }
}

function formatBackendError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes('ECONNREFUSED') || message.includes('Backend not running')) {
    return 'Backend not running. Start python scripts/start_backend.py';
  }
  return message;
}

function getWorkspaceProjectPath(): string | undefined {
  const activeUri = vscode.window.activeTextEditor?.document.uri;
  if (activeUri) {
    const activeFolder = vscode.workspace.getWorkspaceFolder(activeUri);
    if (activeFolder) {
      return activeFolder.uri.fsPath;
    }
  }
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function isSourceListingRequest(text: string): boolean {
  const normalized = text.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return false;
  }
  const patterns = [
    /\blist (the )?(source files|sources)\b/,
    /\bshow (the )?(source files|sources)\b/,
    /\bwhich files did you use\b/,
    /\bwhat (source files|sources) did you use\b/,
    /\bwhat (source files|sources) did you use for (this|the) explanation\b/,
    /\blist (the )?sources used\b/,
    /\bsources used\b/
  ];
  return patterns.some((pattern) => pattern.test(normalized));
}

function formatAssistantMemory(response: GenerateResponse): string {
  const parts: string[] = [`Task: ${response.task}`];
  if (response.explanation.trim()) {
    parts.push(`Explanation:\n${response.explanation.trim()}`);
  }
  if (response.generated_code.trim()) {
    parts.push(`Generated code:\n${response.generated_code.trim()}`);
  }
  if (Array.isArray(response.rag_sources) && response.rag_sources.length > 0) {
    const sources = response.rag_sources.slice(0, 8).map((source) => {
      const metadata = source.metadata && typeof source.metadata === 'object' ? source.metadata : {};
      const relativePath = String(metadata.relative_file_path || metadata.relative_path || source.file_path || 'unknown');
      const symbol = source.symbol_name ? ` symbol=${source.symbol_name}` : '';
      const lines =
        source.start_line || source.end_line ? ` lines=${source.start_line ?? '?'}-${source.end_line ?? '?'}` : '';
      return `- ${relativePath}${symbol}${lines}`;
    });
    parts.push(`Sources:\n${sources.join('\n')}`);
  }
  return parts.join('\n\n');
}

function trimMemoryEntry(value: string): string {
  if (value.length <= MAX_CHAT_HISTORY_ENTRY_CHARS) {
    return value;
  }
  return `${value.slice(0, MAX_CHAT_HISTORY_ENTRY_CHARS).trimEnd()}\n...[trimmed]`;
}
