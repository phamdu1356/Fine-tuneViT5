#!/usr/bin/env python3
"""
patch_venv.py — Áp dụng patches tương thích cho môi trường hiện tại.

Patches:
  1. transformers T5Tokenizer: fix 'dict' vocab → list-of-tuples cho vit5-base
     (bug: tokenizers.Unigram không nhận dict trong transformers >= 5.x)

Cách dùng:
  python patch_venv.py          # kiểm tra và áp dụng nếu cần
  python patch_venv.py --check  # chỉ kiểm tra, không sửa
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


PATCH_MARKER = "# Patch: tokenizers.Unigram requires List[Tuple[str, float]], not dict"


def find_t5_tokenizer() -> Path | None:
    try:
        import transformers
        base = Path(transformers.__file__).parent
        p = base / "models" / "t5" / "tokenization_t5.py"
        return p if p.exists() else None
    except ImportError:
        return None


def needs_patch(path: Path) -> bool:
    return PATCH_MARKER not in path.read_text(encoding="utf-8")


def apply_patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    old = "        if vocab is not None:\n            self._vocab_scores = vocab"
    new = (
        "        if vocab is not None:\n"
        f"            {PATCH_MARKER}\n"
        "            # vit5-base uses legacy format where vocab may be a dict {token: score}\n"
        "            if isinstance(vocab, dict):\n"
        "                self._vocab_scores = list(vocab.items())\n"
        "            else:\n"
        "                self._vocab_scores = vocab"
    )
    if old not in text:
        print(f"[WARN] Pattern not found in {path}. Already patched or file changed.")
        return False
    patched = text.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print("=== patch_venv.py ===")

    t5_path = find_t5_tokenizer()
    if t5_path is None:
        print("[ERROR] transformers not found. Install requirements first.")
        sys.exit(1)

    print(f"T5 tokenizer: {t5_path}")

    if needs_patch(t5_path):
        if args.check:
            print("[NEED PATCH] T5Tokenizer vocab dict bug — run without --check to fix.")
            sys.exit(1)
        else:
            ok = apply_patch(t5_path)
            if ok:
                print("[OK] Patch applied successfully.")
            else:
                print("[WARN] Patch could not be applied — may already be patched differently.")
    else:
        print("[OK] Already patched.")

    print("Done.")


if __name__ == "__main__":
    main()
