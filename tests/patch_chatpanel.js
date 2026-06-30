const fs = require('fs');
let content = fs.readFileSync('vscode-extension/src/chatPanel.ts', 'utf8');

const t1 = \    if (hasCode) {
      bubble.appendChild(createCodeBlock(response.generated_code, response.language, { primary: true, responseId: payload.responseId }));
    }\;

const r1 = \    if (response.edits && response.edits.length > 0) {
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
      bubble.appendChild(createCodeBlock(response.generated_code, response.language, { primary: true, responseId: payload.responseId }));
    }\;

const t2 = \    if (hasCode) {
      var applyBtn = button('Apply changes', 'apply', '');
      applyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'apply', responseId: payload.responseId }); });
      actions.appendChild(applyBtn);
    }\;

const r2 = \    if (response.edits && response.edits.length > 0) {
      var applyAllBtn = button('Apply All Edits', 'apply', '');
      applyAllBtn.addEventListener('click', function () { vscode.postMessage({ type: 'applyAll', responseId: payload.responseId }); });
      actions.appendChild(applyAllBtn);
    } else if (hasCode) {
      var applyBtn = button('Apply changes', 'apply', '');
      applyBtn.addEventListener('click', function () { vscode.postMessage({ type: 'apply', responseId: payload.responseId }); });
      actions.appendChild(applyBtn);
    }\;

if (content.includes(t1)) {
  content = content.replace(t1, r1);
  console.log('Replaced t1 successfully');
} else {
  console.log('t1 not found!');
}

if (content.includes(t2)) {
  content = content.replace(t2, r2);
  console.log('Replaced t2 successfully');
} else {
  console.log('t2 not found!');
}

fs.writeFileSync('vscode-extension/src/chatPanel.ts', content, 'utf8');

