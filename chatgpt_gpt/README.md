# Свой GPT в ChatGPT

Нужна подписка с доступом к созданию GPT (Plus и выше).

## Что понадобится

Из архива `chatgpt-gpt-files.zip` на странице Releases (или соберите его сами: `python tools/build_release.py`). Wheel-файлы `onnxruntime` лежат в Releases отдельно; `python tools/build_release.py --wheels` кладёт их прямо в архив.

| Файл | Зачем |
|---|---|
| `GPT_instructions.txt` | текст для поля Instructions |
| `analyze_gel.py` | скрипт анализа |
| `ladders.json` | справочник маркеров |
| `gelgenie_model.zip` | нейросеть GelGenie (float16) в zip, чтобы Knowledge принял файл |
| `NOTICE.md`, `LICENSE-GelGenie.txt` | лицензия и происхождение модели |
| `onnxruntime-*-cp311-*.whl`, `onnxruntime-*-cp312-*.whl` | запасной вариант, если в песочнице нет `onnxruntime` |

## Шаги

1. Explore GPTs → Create → Configure.
2. Name и Description — на ваш вкус, например «Разбор электрофореза».
3. Instructions: вставьте весь текст из `GPT_instructions.txt`.
4. Knowledge: загрузите остальные файлы из таблицы.
5. Capabilities: включите Code Interpreter & Data Analysis.
6. Conversation starters, например: «Разбери мой гель», «Есть ли продукт в ПЦР?», «Что не так с этим гелем?».
7. Сохраните GPT и проверьте его на гелях из папки `test_gels`.

## Если что-то не работает

- **Knowledge не принимает `.zip` или `.whl`.** Загрузите эти файлы прямо в чат вместе с фото геля:
  они попадут в ту же папку `/mnt/data`, и инструкции их найдут.
- **Нет `onnxruntime` и нет подходящего wheel.** Посмотрите версию Python в песочнице (попросите GPT выполнить
  `import sys; print(sys.version)`) и скачайте wheel `onnxruntime` для этой версии (manylinux, x86_64) с PyPI.
- **Нумерация дорожек не совпадает с лунками.** Это ожидаемая ситуация: GPT должен заметить её по картинке с разметкой
  и перезапустить скрипт с `--add-empty-left/right` или `--lane-centers`. Если не заметил, подскажите ему.

Интерфейс ChatGPT меняется; если названия разделов отличаются, ищите поля с тем же смыслом.
