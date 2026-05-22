from backend.tools.formatter import extract_code_and_explanation


def test_formatter_keeps_perf_opt_list_comprehension():
    raw = "result = [item * 2 for item in items]"

    formatted = extract_code_and_explanation(raw, "python")

    assert formatted.code == "result = [item * 2 for item in items]"
    assert formatted.is_empty is False


def test_formatter_extracts_inline_fenced_python_code():
    raw = "```python result = [item * 2 for item in items]```"

    formatted = extract_code_and_explanation(raw, "python")

    assert formatted.code == "result = [item * 2 for item in items]"


def test_formatter_extracts_common_language_fences():
    cases = [
        ("javascript", "```js\nconst total = add(1, 2);\n```", "const total = add(1, 2);"),
        ("typescript", "```ts\nconst total: number = add(1, 2);\n```", "const total: number = add(1, 2);"),
        ("java", "```java\nclass Demo {}\n```", "class Demo {}"),
        ("cpp", "```cpp\nint main() { return 0; }\n```", "int main() { return 0; }"),
        ("csharp", "```csharp\nclass Demo {}\n```", "class Demo {}"),
        ("rust", "```rust\nfn main() {}\n```", "fn main() {}"),
    ]

    for language, raw, expected in cases:
        formatted = extract_code_and_explanation(raw, language)
        assert formatted.code == expected


def test_formatter_drops_leading_prose_for_non_python_code():
    raw = "Here is the refactor:\n\npublic class Demo {\n    public int Value() { return 1; }\n}"

    formatted = extract_code_and_explanation(raw, "java")

    assert formatted.code.startswith("public class Demo")
    assert "Here is" not in formatted.code
