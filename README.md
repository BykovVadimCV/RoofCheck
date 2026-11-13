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

Основной кандидат для дальнейших экспериментов — существующий depth-first рендерер `roofcheck/synth.py`. Для фотографии, уже совмещённой с `reference.json`, используйте `tools/render_existing.py`; подробности и ограничения — в [инструкции](docs/existing-renderer.md).

| Файл | Что делает |
|---|---|
| `roofcheck/synth.py` | Генератор снега на скате. Толщина в мм на метрической сетке выпрямленного ската; количество по СП 20.13330 (μ по уклону, ветровой перенос с сохранением массы, сугробы за трубами длиной 2h), глубины распределены вокруг порога уборки 30 см. Состояния: не убрано, убрано частично (полосы, полоса у карниза), убрано (остаток плёнкой, недочищенные комья), оттепель. Объём: торец снега у карниза, шапки на трубах и ограждении, карнизный свес. Непрозрачность по двухпотоковой модели |
| `roofcheck/normals.py` | Нормали всего кадра (MoGe-2, MIT): направление «вверх» по фасадам, наклон каждой поверхности, уклон кровли. Снег ложится на все поверхности с наклоном меньше 60°: соседние кровли, карнизы, подоконники, землю. Веса: `models/moge-2-vitb-normal.pt` |
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

# Генератор с явными масками (пункты 1–7)

`roofcheck/roof/` — отдельный конвейер для произвольного размеченного фото: существующий снег, неизвестные области, ROI, препятствия, восемь сценариев, контроль площади и отдельная геометрия скатов. Выходная маска строго 0/1/255. Исходные снег и перекрывающие объекты сохраняются после рендеринга. MoGe и модельные веса не требуются.

    python -m roofcheck.roof sample.json --output out/roof-example --seed 42
    python -m unittest discover -s tests -p 'test_roof_generator.py' -v

Контракт входа, примеры геометрии, определения покрытия и ограничения: [docs/roof-generator-v01.md](docs/roof-generator-v01.md).

## V0.2: освещение и объём

По умолчанию новый конвейер рендерит 2.5D-слой со скруглёнными краями, торцами, нормалями и тенями. `height_px` управляет визуальной высотой; метки и доля покрытия рассчитываются после проекции. Метрическую толщину по фото этот режим не измеряет. Старый режим доступен через `render_mode="flat"`.

[Параметры, диагностические карты и сравнение](docs/roof-generator-v02.md).

## V0.3: реализм кромки

Дискретные столбцы заменены непрерывным пересечением луча с полем высот. Края истончаются к металлу, узкие полосы ниже широких пятен, формы менее регулярны. Микрорельеф, фактура тонкого снега и шум в единицах камеры уменьшают эффект гладкой белой вставки. Контракт масок и защита исходных пикселей сохранены.

[Изменения и ограничения V0.3](docs/roof-generator-v03.md).

## Реальные текстуры снега

Для режима `volume` доступен необязательный `texture_path`: текстура влияет на внешний вид при сохранении геометрии и масок. Подключён небольшой фрагмент снега из пользовательского архива и отдельный материал ambientCG. [Настройки, происхождение и ограничения](docs/snow-textures.md).

## Исследование реализма

Сравнение с архивом `snow-survey`, проверка существующего рендерера и источники: [docs/realism-investigation.md](docs/realism-investigation.md). Экспериментальные `panel_residue` и `wind_drift` являются эвристиками; первый требует размеченных `seam_u` и не восстанавливает фальцы автоматически.

[План пилота после пунктов 1–7](docs/pilot-next-steps.md).

Экспериментальная пакетная генерация поверх существующего рендерера и безопасное возобновление описаны в [docs/batch-existing.md](docs/batch-existing.md).
