#!/usr/bin/env python3
"""Aggressively strip Python comments via tokenize (safe: never touches strings).

Preserves: shebang (#!...), encoding decl (coding:/coding=), and functional
directives (type:, noqa, pragma, pylint:, mypy:, pyright:, flake8:, fmt: on/off,
isort:, yapf:, nosec, ruff:). Docstrings are STRING tokens, never removed.

Removes every other comment — both standalone (whole line deleted) and trailing.
Collapses runs of 3+ blank lines to 2. Idempotent.

Usage: strip_py_comments.py [--apply] FILE...
Prints per-file removed-comment counts; with --apply rewrites in place.
"""
import io
import re
import sys
import tokenize

KEEP = re.compile(
    r"(type:|noqa|pragma|pylint:|mypy:|pyright:|flake8:|fmt:\s*(on|off)"
    r"|isort:|yapf:|nosec|ruff:|coding[:=])"
)


def keep_comment(text: str) -> bool:
    return text.startswith("#!") or bool(KEEP.search(text))


def strip(src: str):
    # one comment max per physical line -> map lineno (1-based) -> start col
    removals = {}
    removed = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and not keep_comment(tok.string):
                removals[tok.start[0]] = tok.start[1]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src, -1  # signal: leave untouched

    if not removals:
        return src, 0

    lines = src.splitlines(keepends=False)
    out = []
    for i, line in enumerate(lines, start=1):
        if i in removals:
            col = removals[i]
            prefix = line[:col]
            removed += 1
            if prefix.strip() == "":
                continue  # whole-line comment -> drop the line
            out.append(prefix.rstrip())
        else:
            out.append(line)

    # collapse 3+ consecutive blank lines down to 2
    collapsed = []
    blanks = 0
    for ln in out:
        if ln.strip() == "":
            blanks += 1
            if blanks <= 2:
                collapsed.append(ln)
        else:
            blanks = 0
            collapsed.append(ln)
    result = "\n".join(collapsed)
    if src.endswith("\n"):
        result += "\n"
    return result, removed


def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    files = [a for a in args if a != "--apply"]
    total_files, total_removed, skipped = 0, 0, []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        new, n = strip(src)
        if n == -1:
            skipped.append(path)
            continue
        if n > 0 and new != src:
            total_files += 1
            total_removed += n
            if apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
            else:
                print(f"{n:5d}  {path}")
    print(f"\n{'APPLIED' if apply else 'DRY-RUN'}: {total_removed} comments across {total_files} files")
    if skipped:
        print(f"SKIPPED (tokenize error, left untouched): {len(skipped)}")
        for s in skipped:
            print(f"   ! {s}")


if __name__ == "__main__":
    main()
