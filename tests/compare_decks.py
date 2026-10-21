"""Compares the .k deck of a desktop run with the deck the browser edition exported.

    python tests/compare_decks.py <native_run.py result .json> <deck exported from the browser, .k>

The desktop writes Windows line endings and a run id (r-xxxxxxxx / r-test) into the deck header and title; both are normalised first.
Prints IDENTICAL, or the number of differing lines and the first few.
"""
import json
import re
import sys
from pathlib import Path


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n")
    return re.sub(r"r-(?:[0-9a-f]{8}|test)\b", "r-RUN", text)


def main(result_json: str, browser_deck: str) -> int:
    run = json.load(open(result_json))
    deck = next(Path(run["home"], "outputs", "r-test").glob("mesh_*.k"))
    a = normalise(deck.read_bytes().decode())
    b = normalise(Path(browser_deck).read_bytes().decode())
    if a == b:
        print(f"IDENTICAL ({len(a):,} characters, {a.count(chr(10)):,} lines): {deck.name} vs {Path(browser_deck).name}")
        return 0
    la, lb = a.split("\n"), b.split("\n")
    diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    print(f"DIFFERENT: {len(la):,} vs {len(lb):,} lines, {len(diff):,} differing lines")
    for i in diff[:5]:
        print(f"  line {i + 1}:\n    desktop: {la[i][:100]}\n    browser: {lb[i][:100]}")
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
