"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const chatViewProvider_1 = require("./chatViewProvider");
const commands_1 = require("./commands");
function activate(context) {
    (0, chatViewProvider_1.registerChatView)(context);
    (0, commands_1.registerCommands)(context);
}
function deactivate() {
    // No background process is started by the extension.
}
//# sourceMappingURL=extension.js.map