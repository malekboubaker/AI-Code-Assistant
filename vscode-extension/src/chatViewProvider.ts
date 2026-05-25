import * as path from 'node:path';
import * as vscode from 'vscode';
import { ApiClient, GenerateRequest, GenerateResponse } from './apiClient';
import { AssistantResponsePayload, ChatViewMessage, getChatViewHtml } from './chatPanel';
import { CollectedEditorContext, collectEditorContext, deserializeRange } from './contextCollector';

const CHAT_VIEW_ID = 'aiCodeAssistant.chatView';
const PREVIEW_SCHEME = 'ai-code-assistant-preview';

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
  }

  private async handleMessage(message: ChatViewMessage): Promise<void> {
    try {
      if (message.type === 'send') {
        await this.sendChatRequest(message.text);
        return;
      }
      if (message.type === 'apply') {
        await this.applyGeneratedCode(message.responseId);
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

    const request: GenerateRequest = {
      instruction,
      code: editorContext.code,
      language: editorContext.language,
      task: null,
      file_path: editorContext.filePath,
      project_path: editorContext.projectPath,
      use_rag: useRag
    };

    try {
      const response = await client.generate(request);
      this.lastBackendResponse = response;
      const responseId = this.nextResponseId();
      if (response.generated_code.trim()) {
        this.pendingApply.set(responseId, { context: editorContext, response });
      }
      const payload: AssistantResponsePayload = {
        responseId,
        instruction,
        response,
        contextSummary: editorContext.contextSummary,
        showTiming
      };
      this.postMessage({ type: 'assistant', payload });
    } catch (error) {
      this.postError(formatBackendError(error));
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
