import ast
from pathlib import Path


def test_clone_voice_routes_to_current_direct_tts_model():
    source = Path(__file__).resolve().parents[1].joinpath("gen2b_agent.py").read_text()
    tree = ast.parse(source)
    clone_branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "startswith('clone:')" in ast.unparse(node.test)
    ]
    assert clone_branches
    models = [
        node.value.value
        for branch in clone_branches
        for node in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tts_model" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert "gen2b/tts" in models
