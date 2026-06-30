from backend.rag.dependency_graph import (
    CLASS_INHERITS_CLASS,
    FILE_IMPORTS_FILE,
    FUNCTION_BELONGS_TO_FILE,
    TEST_COVERS_FILE,
    build_dependency_graph,
    extract_routes,
    extract_structure,
)
from backend.rag.project_analyzer import ProjectAnalyzer
from backend.rag.project_map import build_project_map


def test_extract_structure_python_classes_and_functions():
    source = (
        "class AuthService(BaseService):\n"
        "    def login(self):\n"
        "        return True\n\n"
        "def helper():\n"
        "    return 1\n"
    )
    struct = extract_structure(source, "python")
    class_names = [name for name, _bases in struct["classes"]]
    assert "AuthService" in class_names
    auth = next(item for item in struct["classes"] if item[0] == "AuthService")
    assert "BaseService" in auth[1]
    assert "login" in struct["functions"]
    assert "helper" in struct["functions"]


def test_extract_routes_detects_fastapi_handler():
    source = '@router.get("/users")\nasync def list_users():\n    return []\n'
    routes = extract_routes(source, "python")
    assert routes
    assert routes[0]["handler"] == "list_users"
    assert routes[0]["path"] == "/users"
    assert routes[0]["method"] == "GET"


def test_build_dependency_graph_relationships():
    dependency_graph = {
        "auth_service.py": ["user_repository.py", "base_service.py"],
        "test_auth_service.py": ["auth_service.py"],
    }
    structures = {
        "auth_service.py": {"classes": [("AuthService", ["BaseService"])], "functions": ["login"]},
        "base_service.py": {"classes": [("BaseService", [])], "functions": []},
        "user_repository.py": {"classes": [("UserRepository", [])], "functions": ["find_user"]},
        "test_auth_service.py": {"classes": [("TestAuthService", [])], "functions": ["test_login"]},
    }
    reverse: dict[str, int] = {}
    for targets in dependency_graph.values():
        for target in targets:
            reverse[target] = reverse.get(target, 0) + 1

    graph = build_dependency_graph(
        files=list(structures.keys()),
        dependency_graph=dependency_graph,
        reverse_counts=reverse,
        structures=structures,
        calls_by_file={"auth_service.py": ["find_user"]},
        routes_by_file={},
        test_files={"test_auth_service.py"},
        entry_points=[],
        importance={},
        imported_by_counts=reverse,
    )

    edge_types = {edge[2] for edge in graph["edges"]}
    assert FILE_IMPORTS_FILE in edge_types
    assert CLASS_INHERITS_CLASS in edge_types
    assert TEST_COVERS_FILE in edge_types
    assert FUNCTION_BELONGS_TO_FILE in edge_types

    relations = graph["file_relations"]["auth_service.py"]
    assert "user_repository.py" in relations["related_files"]
    assert "base_service.py" in relations["related_files"]
    assert "test_auth_service.py" in relations["related_tests"]
    assert relations["dependency_count"] >= 2
    assert "AuthService" in relations["related_symbols"]
    assert graph["report"]["critical_files"]


def test_project_analyzer_builds_dependency_graph(tmp_path):
    (tmp_path / "base_service.py").write_text(
        "class BaseService:\n    def run(self):\n        return 1\n", encoding="utf-8"
    )
    (tmp_path / "user_repository.py").write_text(
        "class UserRepository:\n    def find_user(self, uid):\n        return uid\n", encoding="utf-8"
    )
    (tmp_path / "auth_service.py").write_text(
        "from base_service import BaseService\n"
        "from user_repository import UserRepository\n\n"
        "class AuthService(BaseService):\n"
        "    def __init__(self):\n"
        "        self.repo = UserRepository()\n"
        "    def login(self, uid):\n"
        "        return self.repo.find_user(uid)\n",
        encoding="utf-8",
    )
    (tmp_path / "test_auth_service.py").write_text(
        "from auth_service import AuthService\n\n"
        "class TestAuthService:\n"
        "    def test_login(self):\n"
        "        assert AuthService().login(1) == 1\n",
        encoding="utf-8",
    )
    files = [
        tmp_path / name
        for name in ["base_service.py", "user_repository.py", "auth_service.py", "test_auth_service.py"]
    ]

    analysis = ProjectAnalyzer().analyze(str(tmp_path), files)
    graph = analysis.graph
    assert graph

    relations = graph["file_relations"]["auth_service.py"]
    assert "user_repository.py" in relations["related_files"]
    assert "base_service.py" in relations["related_files"]
    assert "test_auth_service.py" in relations["related_tests"]
    assert "AuthService" in relations["related_symbols"]

    edge_types = {edge[2] for edge in graph["edges"]}
    assert CLASS_INHERITS_CLASS in edge_types
    assert TEST_COVERS_FILE in edge_types

    project_map = build_project_map(str(tmp_path), files, analysis=analysis)
    summary = project_map.to_summary()
    assert any(label in summary for label in ("Critical files", "Architecture hotspots", "Dependency relationships"))
