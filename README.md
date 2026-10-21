# RoofCheck

Snow-clearance check for building roofs (Moscow pilot, ООО «Интел-Сет»). This repository holds the pipeline for
photos taken from a taller neighbouring building: registration to a reference frame, overlay masks,
roof-slope rectification, synthetic snow with exact labels, and the validation tooling.

## Layout

| Path | What |
|---|---|
| `roofcheck/` | Pipeline modules (schema, registration, overlays, rectification, synthetic snow, splits, CVAT I/O) |
| `tests/` | Runnable checks and batch generation (`stage1.py`, `synth_batch.py`, `stage2_sheet.py`, `hf_check.py`) |
| `tools/` | Open-photo harvesting and CLIP relevance filter (run on a local machine) |
| `data/objects/` | Per-object references: only the JSON markup is in git |
| `docs/` | Research notes and decisions |
| `legacy/snowtex/` | Earlier parking-lot generator (depth_v4), CC0 texture fetcher, harvest scripts, design-report sources |

## Data (not in git)

Client photos, synthetic batches and model weights stay on disk; see `.gitignore`.

- `data/raw/` — report photos (file names as in `tests/stage1.py`)
- `data/objects/<id>/vantages/<v>/reference.jpg` — reference frame next to its `reference.json`
- `data/textures/` — CC0 snow scans (Poly Haven / ambientCG, `legacy/snowtex/free/fetch_cc0.py`)
- `out/` — generated output

# Этап 1: съёмка с высотных зданий

Снимок → привязка к эталону точки съёмки → перенос контура кровли → маски наложений → выпрямление ската.
Нейросети снега здесь ещё нет (этап 3).

## Модули

| Файл | Что делает |
|---|---|
| `roofcheck/schema.py` | Структура данных: объект → точка съёмки → эталон → снимки; классы разметки |
| `roofcheck/cvat.py` | Набор меток для CVAT, экспорт/импорт контуров в формате CVAT for images 1.1 |
| `roofcheck/customer_markup.py` | Красный контур заказчика → многоугольник кровли и 4 угла ската |
| `roofcheck/register.py` | Привязка: ALIKED + LightGlue (kornia) + USAC_MAGSAC, уточнение по окрестности кровли, проверки качества |
| `roofcheck/overlays.py` | Маски метки времени и значка GPS, чтение даты (RapidOCR), статус GPS по цвету значка |
| `roofcheck/rectify.py` | Выпрямление ската, площадь каждого пикселя, уточнение боковых сторон ската по фальцам, проверка параллельности фальцев |
| `tests/stage1.py` | Прогон всех проверок → `out/stage1_report.json`, `out/stage1_*.jpg` |

## Запуск

    pip install -r requirements.txt
    python3 tests/stage1.py

Веса ALIKED и LightGlue скачиваются при первом запуске (релизы GitHub).

## Лицензии зависимостей

kornia, LightGlue — Apache-2.0; ALIKED — BSD-3; OpenCV — Apache-2.0; RapidOCR — Apache-2.0; PyTorch — BSD.
SuperPoint не используется (ограничительная лицензия).

# Этап 2: данные и синтетика

| Файл | Что делает |
|---|---|
| `roofcheck/synth.py` | Генератор снега на скате: толщина в мм на метрической сетке выпрямленного ската, 8 сценариев, балансировка доли покрытия, снег на соседних кровлях, экспозиция телефона, наложения поверх |
| `tests/synth_batch.py N` | N образцов на каждый фон точки съёмки → `out/synth/` (снимок, толщина uint16 в 0,1 мм, карта классов, JSON) |
| `tests/stage2_sheet.py` | Контактный лист и проверка «яркость снега / яркость металла» |
| `roofcheck/split.py` | Деление реальных снимков по объектам, замороженный тест |
| `LABELING.md` | Правила разметки |
| `tools/harvest_photos.py --set roofs` | Сбор открытых фото кровель (CC0 / CC BY) на своём компьютере → `photos_roofs/` |
| `tools/clip_filter.py` | Отбор по сцене, режим `mode="roofs"` |

Сценарии: fresh, drifted, cleared_eave_band, cleared_strips, residual_film, ridge_melt, trodden_path, cornice.
