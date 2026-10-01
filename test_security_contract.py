"""Check the real entrypoint installs the default gate without importing paid providers."""
import ast
from pathlib import Path
import unittest

class EntrypointTests(unittest.TestCase):
    def test_main_installs_gate_on_only_app(self):
        tree = ast.parse(Path(__file__).with_name("main.py").read_text())
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "service_auth"]
        self.assertTrue(any(a.name == "ServiceAuthMiddleware" and a.asname is None for n in imports for a in n.names))
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "app" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "FastAPI()")
        gates = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "app.add_middleware"]
        self.assertEqual(len(gates), 1)
        self.assertEqual([ast.unparse(x) for x in gates[0].value.args], ["ServiceAuthMiddleware"])
        self.assertGreater(gates[0].lineno, assignments[0].lineno)

    def test_credentials_have_no_nonempty_literal_defaults(self):
        tree = ast.parse(Path(__file__).with_name("main.py").read_text())
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute) or n.func.attr != "getenv" or len(n.args) < 2:
                continue
            key = getattr(n.args[0], "value", "")
            if isinstance(key, str) and any(x in key.upper() for x in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
                self.assertIn(getattr(n.args[1], "value", object()), (None, ""), "Credential fallback at line %s" % n.lineno)

if __name__ == "__main__":
    unittest.main()
