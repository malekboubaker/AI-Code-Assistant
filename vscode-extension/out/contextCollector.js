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
exports.getChatContextOptions = getChatContextOptions;
exports.collectEditorContext = collectEditorContext;
exports.deserializeRange = deserializeRange;
const vscode = __importStar(require("vscode"));
function getChatContextOptions() {
    const config = vscode.workspace.getConfiguration('aiCodeAssistant');
    return {
        includeCursorContext: config.get('chat.includeCursorContext', true),
        maxContextLines: Math.max(1, config.get('chat.maxContextLines', 80))
    };
}
function collectEditorContext(options = getChatContextOptions()) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return {
            code: '',
            language: 'plaintext',
            hasSelection: false,
            contextSummary: 'No active editor. Sending instruction and workspace context only.',
            projectPath: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
        };
    }
    const document = editor.document;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
    const filePath = document.uri.scheme === 'file' ? document.uri.fsPath : undefined;
    const base = {
        language: normalizeLanguage(document.languageId),
        filePath,
        projectPath: workspaceFolder?.uri.fsPath ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
        documentUri: document.uri.toString()
    };
    if (!editor.selection.isEmpty) {
        return {
            ...base,
            code: document.getText(editor.selection),
            hasSelection: true,
            selection: serializeRange(editor.selection),
            contextSummary: `Selected code from ${filePath ?? document.fileName}.`
        };
    }
    if (!options.includeCursorContext) {
        return {
            ...base,
            code: '',
            hasSelection: false,
            contextSummary: `No selection. Sending file path and workspace path for ${filePath ?? document.fileName}.`
        };
    }
    const cursorContext = getCursorContext(document, editor.selection.active, options.maxContextLines);
    return {
        ...base,
        code: cursorContext.text,
        hasSelection: false,
        contextSummary: `No selection. Sending ${cursorContext.startLine + 1}-${cursorContext.endLine + 1} from ${filePath ?? document.fileName}.`
    };
}
function deserializeRange(range) {
    return new vscode.Range(new vscode.Position(range.startLine, range.startCharacter), new vscode.Position(range.endLine, range.endCharacter));
}
function getCursorContext(document, position, maxContextLines) {
    if (document.lineCount <= maxContextLines) {
        const lastLine = Math.max(0, document.lineCount - 1);
        return {
            text: document.getText(),
            startLine: 0,
            endLine: lastLine
        };
    }
    const halfWindow = Math.floor(maxContextLines / 2);
    let startLine = Math.max(0, position.line - halfWindow);
    let endLine = Math.min(document.lineCount - 1, startLine + maxContextLines - 1);
    startLine = Math.max(0, endLine - maxContextLines + 1);
    const range = new vscode.Range(new vscode.Position(startLine, 0), document.lineAt(endLine).range.end);
    return {
        text: document.getText(range),
        startLine,
        endLine
    };
}
function serializeRange(range) {
    return {
        startLine: range.start.line,
        startCharacter: range.start.character,
        endLine: range.end.line,
        endCharacter: range.end.character
    };
}
function normalizeLanguage(languageId) {
    const map = {
        javascript: 'javascript',
        javascriptreact: 'javascript',
        typescript: 'typescript',
        typescriptreact: 'typescript',
        java: 'java',
        cpp: 'cpp',
        c: 'cpp',
        csharp: 'csharp',
        cs: 'csharp',
        python: 'python',
        rust: 'rust'
    };
    return map[languageId] ?? languageId;
}
//# sourceMappingURL=contextCollector.js.map