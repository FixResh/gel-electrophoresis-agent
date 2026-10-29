# Атрибуция

**Модель сегментации:** GelGenie-Universal-FineTune-May-2024 (`unet_dec_21_finetune_epoch_590.onnx`),
https://huggingface.co/mattaq/GelGenie-Universal-FineTune-May-2024
Copyright 2024 University of Edinburgh. Licensed under the Apache License, Version 2.0 (см. `LICENSE-GelGenie.txt`).

**Изменение:** веса сконвертированы из float32 в float16 (onnxconverter-common, `keep_io_types=True`) для уменьшения
размера файла до <30 МБ. На 14 тестовых/валидационных гелях GelGenie маски fp16 и fp32 совпадают на 99.97–99.99 % (Dice).

**Статья:** Aquilina M. et al. GelGenie: an AI-powered framework for gel electrophoresis image analysis.
Nature Communications 16, 2025. https://doi.org/10.1038/s41467-025-59189-0

**Код GelGenie:** https://github.com/mattaq31/GelGenie (Apache-2.0). Предобработка в `analyze_gel.py`
(перевод в оттенки серого, нормализация на 255/65535, паддинг до кратного 32) повторяет `python-gelgenie`.

Скрипт `analyze_gel.py` (разбиение на дорожки, калибровка по маркеру, флаги) написан для этого агента и не является частью GelGenie.
