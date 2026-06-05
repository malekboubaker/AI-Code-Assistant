import * as vscode from 'vscode';

export interface SerializedRange {
  startLine: number;
  startCharacter: number;
  endLine: number;
  endCharacter: number;
}

export interface CollectedEditorContext {
  code: string;
  language: string;
  filePath?: string;
  projectPath?: string;
  documentUri?: string;
  hasSelection: boolean;
  selection?: SerializedRange;
  surroundingContext?: string;
  contextSummary: string;
}

export interface ContextCollectorOptions {
  includeCursorContext: boolean;
  maxContextLines: number;
}

export function getChatContextOptions(): ContextCollectorOptions {
  const config = vscode.workspace.getConfiguration('aiCodeAssistant');
  return {
    includeCursorContext: config.get<boolean>('chat.includeCursorContext', true),
    maxContextLines: Math.max(1, config.get<number>('chat.maxContextLines', 80))
  };
}

export function collectEditorContext(options: ContextCollectorOptions = getChatContextOptions()): CollectedEditorContext {
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
    const surroundingContext = options.includeCursorContext
      ? getSelectionSurroundingContext(document, editor.selection, Math.min(options.maxContextLines, 40)).text
      : '';
    return {
      ...base,
      code: document.getText(editor.selection),
      hasSelection: true,
      selection: serializeRange(editor.selection),
      surroundingContext,
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
    contextSummary: `No selection. Sending ${cursorContext.startLine + 1}-${cursorContext.endLine + 1} from ${
      filePath ?? document.fileName
    }.`
  };
}

export function deserializeRange(range: SerializedRange): vscode.Range {
  return new vscode.Range(
    new vscode.Position(range.startLine, range.startCharacter),
    new vscode.Position(range.endLine, range.endCharacter)
  );
}

function getCursorContext(
  document: vscode.TextDocument,
  position: vscode.Position,
  maxContextLines: number
): { text: string; startLine: number; endLine: number } {
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

  const range = new vscode.Range(
    new vscode.Position(startLine, 0),
    document.lineAt(endLine).range.end
  );
  return {
    text: document.getText(range),
    startLine,
    endLine
  };
}

function getSelectionSurroundingContext(
  document: vscode.TextDocument,
  selection: vscode.Selection,
  maxContextLines: number
): { text: string; startLine: number; endLine: number } {
  const selectedLineCount = selection.end.line - selection.start.line + 1;
  const extraLines = Math.max(2, Math.floor((maxContextLines - selectedLineCount) / 2));
  const startLine = Math.max(0, selection.start.line - extraLines);
  const endLine = Math.min(document.lineCount - 1, selection.end.line + extraLines);
  const range = new vscode.Range(
    new vscode.Position(startLine, 0),
    document.lineAt(endLine).range.end
  );
  return {
    text: document.getText(range),
    startLine,
    endLine
  };
}

function serializeRange(range: vscode.Range): SerializedRange {
  return {
    startLine: range.start.line,
    startCharacter: range.start.character,
    endLine: range.end.line,
    endCharacter: range.end.character
  };
}

function normalizeLanguage(languageId: string): string {
  const map: Record<string, string> = {
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
