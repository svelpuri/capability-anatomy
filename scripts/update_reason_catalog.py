#!/usr/bin/env python3
"""Regenerate the shipped fixed-reason index without importing plugins."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fixed_reason_values(expression):
    """Extract only literal branches of an explicitly reason-valued expression."""
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        yield expression.value
    elif isinstance(expression, ast.IfExp):
        yield from _fixed_reason_values(expression.body)
        yield from _fixed_reason_values(expression.orelse)
    elif (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
          and expression.func.id == 'getattr' and len(expression.args) == 3):
        yield from _fixed_reason_values(expression.args[2])


def reason_catalog(root=ROOT):
    reasons = {}
    for path in sorted([*(root / 'src/capability_anatomy').rglob('*.py'), root / 'scripts/export_standalone.py']):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in {'reason', 'success_reason'}:
                for reason in _fixed_reason_values(node.value):
                    reasons.setdefault(reason, set()).add(path.stem)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                names = [target.id if isinstance(target, ast.Name) else target.attr
                         for target in targets if isinstance(target, (ast.Name, ast.Attribute))]
                if 'reason' in names:
                    for reason in _fixed_reason_values(value):
                        reasons.setdefault(reason, set()).add(path.stem)
                if isinstance(value, ast.Dict) and set(names) & {'_REASONS', '_PUBLIC_ERRORS'}:
                    for key in value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            reasons.setdefault(key.value, set()).add(path.stem)
                for target in targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id == '_PUBLIC_ERRORS' and isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str)):
                        reasons.setdefault(target.slice.value, set()).add(path.stem)
            if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id in {'reason', '_reason'} and isinstance(node.iter, (ast.Tuple, ast.List))):
                for item in node.iter.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        reasons.setdefault(item.value, set()).add(path.stem)
    return reasons


def render():
    header = '''# Fixed reason index

This index covers statically declared operation decisions and typed storage,
evidence and scoring refusals. It is generated from the distributed source by
`python scripts/update_reason_catalog.py`; source module names identify the
consumer to inspect. A reason is a classification, not exception text. Inspect
its correlated span status, duration, decision event and the command's safe
message to distinguish a refusal from a completed operation. Runtime plugins may
emit additional bounded reason names; they are responsible for documenting them.

See [operations](operations.md) for recovery, exit codes and resource limits,
and [observability](observability.md) for collector queries. Do not treat a
historical `complete` state as proof that the current invocation or telemetry
delivery succeeded. Keep the entire evidence directory and adjacent receipts.

| Reason | Source module |
| --- | --- |
'''
    return header + ''.join(f'| `{reason}` | {", ".join(sorted(modules))} |\n' for reason, modules in sorted(reason_catalog().items()))


if __name__ == '__main__':
    (ROOT / 'docs/reasons.md').write_text(render())
