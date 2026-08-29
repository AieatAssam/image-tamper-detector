#!/usr/bin/env python3
"""
Validate the plan itself. Run this before executing any stage, and again after
editing any plan file.

    python3 plan/validate.py

Checks:
  1. every plan/**/*.yaml parses
  2. every stage file referenced by plan.yaml exists and its id matches
  3. every stage carries the keys plan.yaml's stage_file_schema marks required
  4. depends_on agrees between plan.yaml and the stage file, and names real ids
  5. the DAG is acyclic; prints the execution order
  6. every acceptance entry has both `check` and `command`
  7. no benchmark_report (Tier B) entry carries a `command` -- Tier B must never gate
  8. every shell snippet in `acceptance[*].command` and `steps[*].commands`
     parses under `bash -n`

Requires PyYAML. Exit code 0 means the plan is internally consistent; it says
nothing about whether the code is correct.
"""
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    plan_path = os.path.join(ROOT, "plan", "plan.yaml")
    plan = yaml.safe_load(open(plan_path))
    stages = plan["stages"]
    ids = {s["id"] for s in stages}
    required = [k for k, v in plan["stage_file_schema"].items() if v.get("required")]
    errors: list[str] = []
    n_shell = 0

    def bash_n(script: str, where: str) -> None:
        nonlocal n_shell
        n_shell += 1
        r = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        if r.returncode != 0:
            last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"
            errors.append(f"SHELL SYNTAX {where}: {last}")

    for stage in stages:
        path = os.path.join(ROOT, "plan", stage["file"])
        if not os.path.exists(path):
            errors.append(f"missing stage file: {path}")
            continue
        d = yaml.safe_load(open(path))
        if d.get("id") != stage["id"]:
            errors.append(f"{path}: id {d.get('id')!r} != {stage['id']!r}")
        for key in required:
            if key not in d:
                errors.append(f"{path}: missing required key {key!r}")
        if d.get("depends_on") != stage["depends_on"]:
            errors.append(
                f"{path}: depends_on {d.get('depends_on')} disagrees with plan.yaml {stage['depends_on']}"
            )
        for dep in d.get("depends_on", []):
            if dep not in ids:
                errors.append(f"{path}: depends on unknown stage {dep!r}")
        for a in d.get("acceptance", []):
            if "check" not in a or "command" not in a:
                errors.append(f"{path}: acceptance entry lacks check/command")
                continue
            bash_n(a["command"], f"{stage['id']} acceptance {a['check'][:45]!r}")
        for st in d.get("steps", []):
            if isinstance(st.get("commands"), str):
                bash_n(st["commands"], f"{stage['id']} step {st.get('n')}")
        for b in d.get("benchmark_report") or []:
            if "command" in b:
                errors.append(f"{path}: Tier B benchmark_report entry has a command; it must not gate")

    seen: set[str] = set()
    order: list[str] = []

    def visit(i: str, stack: tuple[str, ...] = ()) -> None:
        if i in stack:
            errors.append(f"CYCLE: {' -> '.join(stack + (i,))}")
            return
        if i in seen:
            return
        node = next(s for s in stages if s["id"] == i)
        for dep in node["depends_on"]:
            visit(dep, stack + (i,))
        seen.add(i)
        order.append(i)

    for stage in stages:
        visit(stage["id"])

    for ref in plan["reference_files"]:
        if not os.path.exists(os.path.join(ROOT, ref["path"])):
            errors.append(f"missing reference file {ref['path']}")

    print("execution order:", " -> ".join(order))
    print(f"stages: {len(stages)} | shell snippets checked: {n_shell}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(" -", e)
        return 1
    print("\nAll structural and shell-syntax checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
