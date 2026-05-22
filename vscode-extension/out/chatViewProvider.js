"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.registerChatView = registerChatView;
const path = __importStar(require("node:path"));
const vscode = __importStar(require("vscode"));
const apiClient_1 = require("./apiClient");
const chatPanel_1 = require("./chatPanel");
const contextCollector_1 = require("./contextCollector");
const CHAT_VIEW_ID = 'aiCodeAssistant.chatView';
const PREVIEW_SCHEME = 'ai-code-assistant-preview';
function registerChatView(context) {
    const provider = new ChatViewProvider(context);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(CHAT_VIEW_ID, provider, {
        webviewOptions: {
            retainContextWhenHidden: true
        }
    }), vscode.commands.registerCommand('aiCodeAssistant.openChat', async () => {
        await vscode.commands.executeCommand('workbench.view.extension.aiCodeAssistant');
        await vscode.commands.executeCommand(`${CHAT_VIEW_ID}.focus`);
    }));
}
class ChatViewProvider {
    extensionContext;
    webviewView;
    responseCounter = 0;
    pendingApply = new Map();
    previewProvider = new GeneratedCodePreviewProvider();
    constructor(extensionContext) {
        this.extensionContext = extensionContext;
        this.extensionContext.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider(PREVIEW_SCHEME, this.previewProvider));
    }
    resolveWebviewView(webviewView) {
        this.webviewView = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionContext.extensionUri]
        };
        webviewView.webview.html = (0, chatPanel_1.getChatViewHtml)(webviewView.webview);
        webviewView.webview.onDidReceiveMessage((message) => {
            void this.handleMessage(message);
        });
    }
    async handleMessage(message) {
        try {
            if (message.type === 'send') {
                await this.sendChatRequest(message.text);
                return;
            }
            if (message.type === 'apply') {
                await this.applyGeneratedCode(message.responseId);
            }
        }
        catch (error) {
            this.postError(error);
        }
    }
    async sendChatRequest(instruction) {
        const editorContext = (0, contextCollector_1.collectEditorContext)();
        const config = vscode.workspace.getConfiguration('aiCodeAssistant');
        const endpoint = config.get('backendUrl', 'http://localhost:8000/api/v1/generate');
        const timeoutMs = config.get('timeoutMs', 120000);
        const useRag = config.get('chat.useRag', true);
        const showTiming = config.get('chat.showTimingMetadata', false);
        let client;
        try {
            client = new apiClient_1.ApiClient(endpoint, timeoutMs);
        }
        catch (error) {
            this.postError(formatBackendError(error));
            return;
        }
        const request = {
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
            const responseId = this.nextResponseId();
            if (response.generated_code.trim()) {
                this.pendingApply.set(responseId, { context: editorContext, response });
            }
            const payload = {
                responseId,
                instruction,
                response,
                contextSummary: editorContext.contextSummary,
                showTiming
            };
            this.postMessage({ type: 'assistant', payload });
        }
        catch (error) {
            this.postError(formatBackendError(error));
        }
    }
    async applyGeneratedCode(responseId) {
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
    async replaceSelection(context, generatedCode) {
        const document = await vscode.workspace.openTextDocument(vscode.Uri.parse(context.documentUri));
        const editor = await vscode.window.showTextDocument(document, { preview: false });
        await editor.edit((editBuilder) => {
            editBuilder.replace((0, contextCollector_1.deserializeRange)(context.selection), generatedCode);
        });
    }
    async previewAndApplyFullFile(context, generatedCode) {
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
        const previewUri = this.previewProvider.setContent(generatedCode, path.basename(targetUri.fsPath || targetUri.path));
        await vscode.commands.executeCommand('vscode.diff', document.uri, previewUri, `AI Code Assistant Preview: ${path.basename(document.fileName)}`);
        const choice = await vscode.window.showWarningMessage('Preview opened. Apply generated code as a full-file replacement? For targeted edits, select code first.', { modal: true }, 'Apply to File');
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
    nextResponseId() {
        this.responseCounter += 1;
        return `${Date.now()}-${this.responseCounter}`;
    }
    postError(error) {
        this.postMessage({ type: 'error', message: typeof error === 'string' ? error : formatBackendError(error) });
    }
    postMessage(message) {
        void this.webviewView?.webview.postMessage(message);
    }
}
class GeneratedCodePreviewProvider {
    contentByUri = new Map();
    onDidChangeEmitter = new vscode.EventEmitter();
    onDidChange = this.onDidChangeEmitter.event;
    setContent(content, fileName) {
        const uri = vscode.Uri.from({
            scheme: PREVIEW_SCHEME,
            path: `/${fileName}`,
            query: `t=${Date.now()}`
        });
        this.contentByUri.set(uri.toString(), content);
        this.onDidChangeEmitter.fire(uri);
        return uri;
    }
    provideTextDocumentContent(uri) {
        return this.contentByUri.get(uri.toString()) ?? '';
    }
}
function formatBackendError(error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes('ECONNREFUSED') || message.includes('Backend not running')) {
        return 'Backend not running. Start python scripts/start_backend.py';
    }
    return message;
}
//# sourceMappingURL=chatViewProvider.js.map