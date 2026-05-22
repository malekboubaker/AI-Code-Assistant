import * as vscode from 'vscode';
import { registerChatView } from './chatViewProvider';
import { registerCommands } from './commands';

export function activate(context: vscode.ExtensionContext): void {
  registerChatView(context);
  registerCommands(context);
}

export function deactivate(): void {
  // No background process is started by the extension.
}
