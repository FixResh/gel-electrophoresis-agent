#!/usr/bin/env python3
"""Собирает архивы для страницы Releases:

  dist/gel-electrophoresis-analyzer.zip  — скилл для Claude (папка скилла внутри архива)
  dist/chatgpt-gpt-files.zip             — файлы для своего GPT (плоский список для Knowledge)

  python tools/build_release.py            # без wheel-файлов onnxruntime
  python tools/build_release.py --wheels   # плюс офлайн-wheel onnxruntime для Python 3.11 и 3.12 (нужен pip и интернет)
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "claude_skill", "gel-electrophoresis-analyzer")
DIST = os.path.join(ROOT, "dist")
ORT_VERSION = "1.20.1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add(zf, src, arc):
    info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))  # одинаковые даты -> воспроизводимый архив
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with open(src, "rb") as f:
        zf.writestr(info, f.read(), compresslevel=9)


def build_skill():
    out = os.path.join(DIST, "gel-electrophoresis-analyzer.zip")
    files = []
    for dirpath, dirnames, filenames in os.walk(SKILL):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            files.append((full, os.path.join("gel-electrophoresis-analyzer", os.path.relpath(full, SKILL))))
    with zipfile.ZipFile(out, "w") as zf:
        for src, arc in files:
            add(zf, src, arc.replace(os.sep, "/"))
    return out


def download_wheels(dest):
    wheels = []
    for py in ("3.11", "3.12"):
        cmd = [sys.executable, "-m", "pip", "download", f"onnxruntime=={ORT_VERSION}", "--no-deps",
               "--only-binary=:all:", "--python-version", py, "--platform", "manylinux_2_27_x86_64", "-d", dest]
        subprocess.run(cmd, check=True)
    for fn in sorted(os.listdir(dest)):
        if fn.endswith(".whl"):
            wheels.append(os.path.join(dest, fn))
    return wheels


def build_gpt(with_wheels):
    out = os.path.join(DIST, "chatgpt-gpt-files.zip")
    with tempfile.TemporaryDirectory() as tmp:
        model_zip = os.path.join(tmp, "gelgenie_model.zip")
        with zipfile.ZipFile(model_zip, "w") as mz:
            add(mz, os.path.join(SKILL, "model", "gelgenie_unet_fp16.onnx"), "gelgenie_unet_fp16.onnx")
        items = [
            (os.path.join(ROOT, "chatgpt_gpt", "GPT_instructions.txt"), "GPT_instructions.txt"),
            (os.path.join(SKILL, "scripts", "analyze_gel.py"), "analyze_gel.py"),
            (os.path.join(SKILL, "ladders.json"), "ladders.json"),
            (model_zip, "gelgenie_model.zip"),
            (os.path.join(SKILL, "NOTICE.md"), "NOTICE.md"),
            (os.path.join(SKILL, "LICENSE-GelGenie.txt"), "LICENSE-GelGenie.txt"),
        ]
        if with_wheels:
            for w in download_wheels(tmp):
                items.append((w, os.path.basename(w)))
        with zipfile.ZipFile(out, "w") as zf:
            for src, arc in items:
                add(zf, src, arc)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wheels", action="store_true", help="добавить офлайн-wheel onnxruntime в архив для ChatGPT")
    args = p.parse_args()
    os.makedirs(DIST, exist_ok=True)
    for path in (build_skill(), build_gpt(args.wheels)):
        size = os.path.getsize(path) / 1e6
        print(f"{os.path.relpath(path, ROOT)}  {size:.1f} МБ  sha256 {sha256(path)}")
        if size > 30 and path.endswith("gel-electrophoresis-analyzer.zip"):
            print("  ВНИМАНИЕ: больше 30 МБ — Claude может не принять архив")


if __name__ == "__main__":
    main()
