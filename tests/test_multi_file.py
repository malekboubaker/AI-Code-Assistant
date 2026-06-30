from backend.tools.formatter import extract_multi_file_edits

def test_extract_multi_file_edits():
    raw_output = """Here are the changes:
<file path="src/main.py" reason="Updated entrypoint">
def main():
    print("Hello")
</file>

<file path="src/utils.py">
```python
def helper():
    return True
```
</file>
"""
    edits = extract_multi_file_edits(raw_output)
    
    assert len(edits) == 2
    assert edits[0]["file_path"] == "src/main.py"
    assert edits[0]["reason"] == "Updated entrypoint"
    assert edits[0]["new_content"] == "def main():\n    print(\"Hello\")"
    
    assert edits[1]["file_path"] == "src/utils.py"
    assert edits[1]["reason"] == "Refactored by AI"
    assert edits[1]["new_content"] == "def helper():\n    return True"

