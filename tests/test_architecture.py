"""
Static checks of the project structure rules.

- writing_block.md section 1: interface.py is only a connection point (no
  logic, no classes/decorators) and imports from its own implementation.py.
- Components import each other only through interface.py.
- Docs/02 section 2: no out-of-scope frameworks or local clustering library.
- Docs/10 section 3: no print() inside components.

These run against whatever components exist, so they apply automatically
as each person adds their folder.
"""

import ast
from pathlib import Path

import pytest

from conftest import PACKAGE_DIR, REPO_ROOT


PACKAGE = "app_review_intelligence"
FORBIDDEN_IMPORTS = (
    "langchain", "llama_index", "llamaindex",
    "fastapi", "flask", "django", "streamlit", "gradio",
    "chromadb", "faiss", "pinecone", "qdrant_client", "weaviate", "lancedb",
    "sklearn.cluster", "hdbscan", "umap", "docker",
)
# Terminal output belongs to the CLI/reporting layer only.
PRINT_ALLOWED = {"cli", "reporting"}


def _component_dirs():
    return sorted(path.parent for path in PACKAGE_DIR.rglob("interface.py"))


def _component_name(directory):
    return ".".join(directory.relative_to(PACKAGE_DIR).parts)


def _python_files():
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    inference = REPO_ROOT / "inference.py"
    if inference.exists():
        files.append(inference)
    return files


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against the file's package
                package = path.parent.relative_to(PACKAGE_DIR.parent).parts
                base = package[: len(package) - (node.level - 1)]
                yield node, ".".join(base + ((node.module,) if node.module else ()))
            elif node.module:
                yield node, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name


def _owning_component(path, components):
    owners = [c for c in components if c in path.parents or c == path.parent]
    return max(owners, key=lambda c: len(c.parts)) if owners else None


COMPONENT_DIRS = _component_dirs()


@pytest.mark.skipif(not COMPONENT_DIRS, reason="no components exist yet")
@pytest.mark.parametrize("directory", COMPONENT_DIRS, ids=_component_name)
def test_component_has_implementation_and_interface(directory):
    assert (directory / "implementation.py").exists()
    assert (directory / "__init__.py").exists()


@pytest.mark.skipif(not COMPONENT_DIRS, reason="no components exist yet")
@pytest.mark.parametrize("directory", COMPONENT_DIRS, ids=_component_name)
def test_interface_has_no_classes_or_decorators(directory):
    tree = ast.parse((directory / "interface.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        assert not isinstance(node, ast.ClassDef), f"class {node.name} in interface.py"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.decorator_list, f"decorated function {node.name} in interface.py"


@pytest.mark.skipif(not COMPONENT_DIRS, reason="no components exist yet")
@pytest.mark.parametrize("directory", COMPONENT_DIRS, ids=_component_name)
def test_interface_imports_its_own_implementation(directory):
    own = f"{PACKAGE}.{_component_name(directory)}"
    modules = {module for _, module in _imports(directory / "interface.py")}

    assert any(m == f"{own}.implementation" or m.startswith(f"{own}.") for m in modules)


def test_components_import_each_other_only_through_interfaces():
    components = _component_dirs()
    for path in _python_files():
        owner = _owning_component(path, components)
        for node, module in _imports(path):
            if not module.startswith(PACKAGE + "."):
                continue
            target = PACKAGE_DIR.joinpath(*module.split(".")[1:])
            target_owner = _owning_component(target / "x.py", components) or _owning_component(
                target.with_suffix(".py"), components
            )
            if target_owner is None or target_owner == owner:
                continue
            assert module.endswith(".interface"), (
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {module}; "
                "use the other component's interface.py"
            )


def test_no_out_of_scope_dependencies():
    for path in _python_files():
        for node, module in _imports(path):
            for forbidden in FORBIDDEN_IMPORTS:
                assert not (module == forbidden or module.startswith(forbidden + ".")), (
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports out-of-scope {module}"
                )


def test_components_do_not_print():
    for path in PACKAGE_DIR.rglob("*.py"):
        if set(path.relative_to(PACKAGE_DIR).parts) & PRINT_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                pytest.fail(f"{path.relative_to(REPO_ROOT)}:{node.lineno} uses print(); use the logger")


def test_no_hardcoded_api_key_or_key_argument():
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        # Google API keys start with "AIza".
        assert "AIza" not in text, f"{path.relative_to(REPO_ROOT)} looks like it contains an API key"
        # The key is never accepted as a command-line argument (Docs/02 section 8).
        assert "--api-key" not in text and "--gemini-api-key" not in text
