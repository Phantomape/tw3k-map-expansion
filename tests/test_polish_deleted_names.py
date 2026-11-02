"""polish_campaign_terrain runs as one module-level script: a name removed with a top-level `del` must not be read
again further down (v23 R2: `del _memo` at the setup memo, then `_memo is None` in the [TUNDRA OASIS OFF] guard ->
NameError only with TW3K_TUNDRA_OASIS_OFF=1). Static check over the module AST, no run needed."""
import ast
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'polish_campaign_terrain.py'


def reads_after_delete(tree):
    """(name, del line, read line) for every top-level `del x` followed by a load of x with no rebinding between."""
    out = []
    body = tree.body
    for i, st in enumerate(body):
        if not isinstance(st, ast.Delete):
            continue
        for tg in st.targets:
            if not isinstance(tg, ast.Name):
                continue
            n = tg.id
            for later in body[i + 1:]:
                nodes = sorted((x for x in ast.walk(later) if isinstance(x, ast.Name) and x.id == n),
                               key=lambda x: (x.lineno, x.col_offset))
                if not nodes:
                    continue
                if isinstance(nodes[0].ctx, ast.Load):
                    out.append((n, st.lineno, nodes[0].lineno))
                break
    return out


class TestPolishDeletedNames(unittest.TestCase):
    def test_no_read_after_top_level_del(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        self.assertEqual(reads_after_delete(tree), [])

    def test_detects_the_v23_pattern(self):
        tree = ast.parse("_memo=None\ndel _memo\nif True:\n if _memo is None:pass\n")
        self.assertEqual(reads_after_delete(tree), [('_memo', 2, 4)])


if __name__ == '__main__':
    unittest.main()
