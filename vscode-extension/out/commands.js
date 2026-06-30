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
exports.registerCommands = registerCommands;
const vscode = __importStar(require("vscode"));
const apiClient_1 = require("./apiClient");
const COMMANDS = [
    {
        commandId: 'aiCodeAssistant.generateCode',
        task: 'code_gen',
        defaultInstruction: 'Generate code using the selected context. Return only valid code.',
        useRagDefault: false,
        mode: 'insertBelow'
    },
    {
        commandId: 'aiCodeAssistant.fixBug',
        task: 'bug_fix',
        defaultInstruction: 'Fix the bug in the selected code. Return only corrected code.',
        useRagDefault: false,
        mode: 'replaceSelection'
    },
    {
        commandId: 'aiCodeAssistant.refactorSelection',
        task: 'refactoring',
        defaultInstruction: 'Refactor the selected code to be cleaner while preserving behavior. Return only refactored code.',
        useRagDefault: false,
        mode: 'replaceSelection'
    },
    {
        commandId: 'aiCodeAssistant.optimizePerformance',
        task: 'perf_opt',
        defaultInstruction: 'Optimize the selected code for performance while preserving behavior. Return only optimized code.',
        useRagDefault: false,
        mode: 'replaceSelection'
    },
    {
        commandId: 'aiCodeAssistant.generateUnitTests',
        task: 'test_gen',
        defaultInstruction: 'Write complete unit tests for the selected code. Return only valid test code.',
        useRagDefault: false,
        mode: 'insertBelow'
    },
    {
        commandId: 'aiCodeAssistant.explainProjectCode',
        task: 'project_explain',
        defaultInstruction: 'Explain how the selected code works using retrieved project context. Do not generate code.',
        useRagDefault: true,
        mode: 'showExplanation'
    }
];
const output = vscode.window.createOutputChannel('AI Code Assistant');
function registerCommands(context) {
    for (const spec of COMMANDS) {
        context.subscriptions.push(vscode.commands.registerCommand(spec.commandId, async () => runCommand(spec)));
    }
    context.subscriptions.push(output);
}
async function runCommand(spec) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('AI Code Assistant: no active editor.');
        return;
    }
    const selection = editor.selection;
    const selectedCode = editor.document.getText(selection);
    if (!selectedCode.trim()) {
        vscode.window.showErrorMessage('AI Code Assistant: select code before running this command.');
        return;
    }
    const instruction = await vscode.window.showInputBox({
        title: commandTitle(spec),
        prompt: 'Instruction sent to the local backend',
        value: spec.defaultInstruction,
        ignoreFocusOut: true
    });
    if (instruction === undefined) {
        return;
    }
    const config = vscode.workspace.getConfiguration('aiCodeAssistant');
    const endpoint = config.get('backendUrl', 'http://localhost:8000/api/v1/generate');
    const timeoutMs = config.get('timeoutMs', 300000);
    const useRagForCodeTasks = config.get('useRagForCodeTasks', false);
    const useRag = spec.useRagDefault || useRagForCodeTasks;
    let client;
    try {
        client = new apiClient_1.ApiClient(endpoint, timeoutMs);
    }
    catch (error) {
        vscode.window.showErrorMessage(`AI Code Assistant: ${error.message}`);
        return;
    }
    const request = buildRequest(editor, selectedCode, instruction, spec.task, useRag);
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `AI Code Assistant: ${commandTitle(spec)}`,
        cancellable: false
    }, async () => {
        try {
            const response = await client.generate(request);
            await handleResponse(editor, selection, spec, response);
        }
        catch (error) {
            vscode.window.showErrorMessage(`AI Code Assistant: ${error.message}`);
        }
    });
}
function buildRequest(editor, code, instruction, task, useRag) {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    return {
        instruction,
        code,
        language: normalizeLanguage(editor.document.languageId),
        task,
        file_path: editor.document.uri.fsPath,
        project_path: workspaceFolder?.uri.fsPath,
        has_selection: true,
        surrounding_context: getManualSurroundingContext(editor),
        use_rag: useRag
    };
}
async function handleResponse(editor, selection, spec, response) {
    if (spec.mode === 'showExplanation') {
        if (!response.explanation.trim()) {
            throw new Error('Model returned an empty explanation.');
        }
        showExplanation(response);
        return;
    }
    if (!response.generated_code.trim()) {
        const warning = response.validation?.warnings?.join(' ') || 'Model returned empty output.';
        throw new Error(warning);
    }
    if (spec.mode === 'replaceSelection') {
        await editor.edit((editBuilder) => {
            editBuilder.replace(selection, response.generated_code);
        });
    }
    else {
        await editor.edit((editBuilder) => {
            editBuilder.insert(selection.end, `\n${response.generated_code}\n`);
        });
    }
    if (response.validation && response.validation.valid === false) {
        vscode.window.showWarningMessage(`AI Code Assistant: inserted output, but backend validation reported warnings: ${response.validation.warnings.join(' ')}`);
    }
}
function showExplanation(response) {
    output.clear();
    output.appendLine('AI Code Assistant: Explain Project Code');
    output.appendLine('');
    output.appendLine(response.explanation);
    output.appendLine('');
    output.appendLine(`RAG used: ${response.used_rag ? 'yes' : 'no'}`);
    if (response.rag_sources.length > 0) {
        output.appendLine('');
        output.appendLine('Sources:');
        for (const source of response.rag_sources) {
            output.appendLine(`- ${source.file_path ?? 'unknown'}:${source.start_line ?? '?'}-${source.end_line ?? '?'} ` +
                `score=${source.score.toFixed(3)} symbol=${source.symbol_name ?? 'unknown'}`);
        }
    }
    output.show(true);
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
function getManualSurroundingContext(editor) {
    const selection = editor.selection;
    const document = editor.document;
    const selectedLineCount = selection.end.line - selection.start.line + 1;
    const extraLines = Math.max(2, Math.floor((40 - selectedLineCount) / 2));
    const startLine = Math.max(0, selection.start.line - extraLines);
    const endLine = Math.min(document.lineCount - 1, selection.end.line + extraLines);
    const range = new vscode.Range(new vscode.Position(startLine, 0), document.lineAt(endLine).range.end);
    return document.getText(range);
}
function commandTitle(spec) {
    switch (spec.task) {
        case 'code_gen':
            return 'Generate Code';
        case 'bug_fix':
            return 'Fix Bug';
        case 'refactoring':
            return 'Refactor Selection';
        case 'perf_opt':
            return 'Optimize Performance';
        case 'test_gen':
            return 'Generate Unit Tests';
        case 'project_explain':
            return 'Explain Project Code';
    }
}
//# sourceMappingURL=commands.js.map