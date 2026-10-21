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
| `roofcheck/synth.py` | Генератор снега на скате. Толщина в мм на метрической сетке выпрямленного ската; количество по СП 20.13330 (μ по уклону, ветровой перенос с сохранением массы, сугробы за трубами длиной 2h), глубины распределены вокруг порога уборки 30 см. Состояния: не убрано, убрано частично (полосы, полоса у карниза), убрано (остаток плёнкой, недочищенные комья), оттепель. Объём: торец снега у карниза, шапки на трубах и ограждении, карнизный свес. Непрозрачность по двухпотоковой модели |
| `roofcheck/hostmatch.py` | Замер съёмки по фону: размытие и резкость краёв, шум по яркости, дымка, клиппинг, качество JPEG. Снег проходит те же искажения |
| `tests/synth_batch.py N` | N образцов на каждый фон → `out/synth/`: снимок, вход сети без наложений (`_net.jpg`), толщина (uint16, 0,1 мм), классы, зона проверки (`_roi.png`), бинарные метки на порогах 1/2/5/20/300 мм (`_t{T}.png`, 255 = пояс ±1 мм и наложения), JSON с параметрами |
| `tests/stage2_sheet.py` | Контактный лист и проверка «яркость снега / яркость металла» |
| `tests/validate.py [--real DIR]` | Проверка без реального снега: тест нулевой вставки (весь конвейер без снега против исходника, с положительным контролем), проверка меток; с `--real` — сравнение статистик реального и синтетического снега |
| `roofcheck/split.py` | Деление реальных снимков по объектам, замороженный тест |
| `LABELING.md` | Правила разметки |
| `tools/harvest_photos.py --set roofs` | Сбор открытых фото кровель (CC0 / CC BY) на своём компьютере → `photos_roofs/` |
| `tools/clip_filter.py` | Отбор по сцене, режим `mode="roofs"` |

Размеры и уклон кровли общие для всех образцов одного объекта: из `reference.geometry`, если заданы, иначе из случайного приближения, закреплённого за `object_id`. Трубы, вентблоки, тарелки и слуховые окна обводятся в эталоне один раз (роль `structure`): без них снег ложится поверх серых труб.

Реальные снимки при обучении проходят то же финальное пересжатие JPEG, что и синтетика: тест нулевой вставки показывает, что пересжатие немного сглаживает цветовой шум.
