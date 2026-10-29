#!/usr/bin/env python3
"""Скачивает оригинальную модель GelGenie (float32) с Hugging Face, проверяет SHA-256
и конвертирует её в float16 — так получен файл model/gelgenie_unet_fp16.onnx в скилле.

  pip install onnx onnxconverter-common
  python tools/fetch_model.py                      # скачать и сконвертировать
  python tools/fetch_model.py --from-file model.onnx   # если оригинал уже скачан вручную

Байтовая копия может отличаться от файла в репозитории при других версиях onnx/onnxconverter-common;
на результат анализа это не влияет.
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://huggingface.co/mattaq/GelGenie-Universal-FineTune-May-2024/resolve/main/"
       "onnx_checkpoints/unet_dec_21_finetune_epoch_590.onnx")
SHA256_FP32 = "df9e39157eda6e1d1f034f5ca75fb2b8d0965189c026f00504a80c2743bb6336"
DEFAULT_OUT = os.path.join(ROOT, "claude_skill", "gel-electrophoresis-analyzer", "model", "gelgenie_unet_fp16.onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-file", help="путь к уже скачанному unet_dec_21_finetune_epoch_590.onnx")
    p.add_argument("--out", default=DEFAULT_OUT, help="куда сохранить модель float16")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        src = args.from_file
        if not src:
            src = os.path.join(tmp, "gelgenie_fp32.onnx")
            print(f"Скачиваю {URL}")
            urllib.request.urlretrieve(URL, src)
        got = sha256(src)
        if got != SHA256_FP32:
            sys.exit(f"SHA-256 не совпадает: {got}\nожидался {SHA256_FP32}")
        print("SHA-256 оригинала совпадает")

        import onnx
        from onnxconverter_common import float16
        model = onnx.load(src)
        model16 = float16.convert_float_to_float16(model, keep_io_types=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        onnx.save(model16, args.out)
    print(f"Сохранено: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} МБ), sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
