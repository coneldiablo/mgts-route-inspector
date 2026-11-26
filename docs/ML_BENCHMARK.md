# Проверка моделей для осмотра люков

## Какая модель за что отвечает

YOLO26n ищет область люка по полному кадру. Детектор дообучен с официальных весов COCO на публичных снимках с проверенными рамками. DINOv2 Small извлекает визуальные признаки из области, а признаки цвета, текстуры и формы дополняют их перед классификацией. Модель различает видимый открытый проём и закрытый люк. Если рамка есть в таблице, классификатор использует её напрямую. [YOLO26](https://docs.ultralytics.com/models/yolo26)

В режиме `--single-target` предполагается, что на каждом снимке есть один целевой люк. Если локализатор дал несколько рамок, выбирается лучшая. При отсутствии рамки можно включить отдельно обученный классификатор полного кадра. Источник вывода и порог видны в результате. В обычном режиме кадры без рамки или с несколькими кандидатами идут на сверку.

Флаг `--defect-only` предназначен для размеченного пакета подтверждённых дефектов с несколькими видами. Порог выбирается по macro-F1 на групповой валидации, отказ снижает полноту класса. При низкой уверенности модель запрашивает уточнение вида.

При явном включении Gemini 4 Argon через Vertex AI перепроверяет кадры, для которых локальная модель вернула слабый сигнал или использовала весь кадр. При отсутствии локальной рамки Argon предлагает свою. Каскад проверяет качество этого кропа, прогоняет DINOv2 повторно и принимает рамку при согласии моделей по видимому состоянию. Расхождение и ошибки запроса записываются в карточку. Для классификации с готовой рамкой оценка Argon служит дополнительным комментарием. [Gemini 4 Argon](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-4-argon)

## Что измерялось

Использованы публичные наборы [CubeAI](https://www.kaggle.com/datasets/cubeai/manhole-for-yolov8) и [Deep Pratap](https://www.kaggle.com/datasets/deeppratap/manhole-detection). В проверочной части 31 фото из Cube и 13 из второго источника после ручного просмотра. Подборки очищены от очевидных повторов и нерелевантных кадров. У источников отсутствуют идентификаторы маршрутов и физических объектов. Подготовка воспроизводится через `scripts/fetch_labeled_defects.py`.

| Эксперимент | Результат на 44 фото |
|---|---:|
| MobileNet + CV, рамка из разметки | 43 верных |
| DINOv2 + CV, рамка из разметки | 44 верных |
| SSDLite → DINOv2 + CV, лучшая рамка | 37 верных, 7 на сверку |
| SSDLite → DINOv2 + CV, максимум одна найденная рамка | 27 верных, 17 на сверку |
| SSDLite → DINOv2 + CV с запасной моделью полного кадра при одном целевом люке | 44 верных. Для 7 фото вывод сделан по полному кадру |
| YOLO26n v1 → DINOv2 + CV, лучшая рамка | 41 верный, 3 на сверку |
| YOLO26n v1 → DINOv2 + CV с запасной моделью полного кадра | 44 верных. Для 3 фото вывод сделан по полному кадру |
| YOLO26n v2 → DINOv2 + CV, лучшая рамка | 35 верных, 9 на сверку |
| YOLO26n v2 → DINOv2 + CV с запасной моделью полного кадра | 44 верных. Для 9 фото вывод сделан по полному кадру |
| Gemini 3.5 Flash через ADC, рамка из разметки | 43 верных, 1 ответ не получен после повтора |

Отдельно проверены рамки на 44 кадрах с люком и ложные рамки на 20 публичных кадрах с визуально подтверждённым отсутствием люка. Среди отвлекающих объектов были колесо и знак. Метрики переноса на маршруты МГТС предстоит измерить отдельно.

| Детектор | Точная верхняя рамка, IoU ≥ 0,5 | Кадры без люка с ложной рамкой |
|---|---:|---:|
| SSDLite | 29/44 | 3/20 |
| YOLO26n v1 | 35/44 | 16/20 |
| YOLO26n v1 после установки порога с учётом фонов | 24/44 | 2/20 |
| YOLO26n v2 с обучением на фонах | 31/44 | 0/20 |

В обучение YOLO26n v2 добавлено 124 проверенных кадра без люка. Порог 0,45 выбран отдельно на 26 кадрах с люком и 25 без него по максимальной полноте рамок при доле ложных срабатываний до 10%. Внутри этого разбиения v2 дал 20 точных рамок из 26 и 0 ложных рамок из 25. На той же валидации для v1 правило выбрало порог 0,80. Для демо выбран v2. [SSDLite](../artifacts/ml-cube-locator-verified/cascade.json) · [YOLO26n v1](../artifacts/ml-yolo26n-verified/report.json) · [YOLO26n v2](../artifacts/ml-yolo26n-background-v2/report.json) · [Каскад v2](../artifacts/ml-yolo26n-background-v2/cascade_44.json)

Стресс-тест повторил классификацию с исходной рамкой после затемнения до 45% яркости, размытия и уменьшения кадра вдвое. DINOv2 + CV дал 44 из 44 в каждом из этих трёх сценариев.

Через Vertex AI проверены одинаковые семь полных кадров, на которых SSDLite не дал рамку. Gemini 4 Argon верно определил состояние на всех семи. На пяти совпали класс и рамка при IoU ≥ 0,5. Gemini 3.5 Flash верно определил состояние на шести и дал точную рамку с классом на четырёх. Один запрос к Flash завершился сетевой ошибкой. Время ответа Argon на этих кадрах составило 5,9–35,1 с. [Отчёт Argon](../artifacts/ml-vertex-argon-full-misses/summary.json) · [Отчёт Flash](../artifacts/ml-vertex-35-full-misses/summary.json)

В интегрированном прогоне v2 Argon получил первые шесть из девяти кадров, оставшихся без рамки YOLO. Четыре предложения прошли проверку состояния локальной моделью по кропу. Три из них пересеклись с разметкой при IoU ≥ 0,5. Два запроса завершились по тайм-ауту, оба кадра сохранили правильный вывод локальной модели. Класс сверили с разметкой по ID для всех 44 фото. Итог прогона — 44 верных состояния, 34 точные рамки с учётом Argon. [Интегрированный прогон](../runs/ml-yolo26n-background-v2-argon/summary.json)

## Откуда взяты решения

- Из исследования состояния крышек взято разделение поиска объекта и классификации области. В экспериментах авторов использовались кропы, поэтому у нас отдельно измерены готовая рамка и полный кадр. [Статья Sensors](https://www.mdpi.com/1424-8220/25/13/4144)
- Для обучаемой части взяты замороженные признаки DINOv2 и классификатор поверх них. Кандидаты обучены на одинаковых данных с порогами из обучающих разбиений. [Работа Meta](https://arxiv.org/abs/2304.07193)
- Из полевого примера FirstEnergy взят принцип измерять найденные дефекты и нагрузку на проверяющих. Для плохого кадра и отсутствующей рамки предусмотрены явные исходы. [Отчёт FirstEnergy](https://www.firstenergycorp.com/newsroom/featured_stories/cameras-with-ai-aim-to-enhance-electric-service-reliability.html)

## Локальный запуск

Распакуйте `trained-models.zip` из [релиза v0.1.0](https://github.com/coneldiablo/mgts-route-inspector/releases/tag/v0.1.0) в корне репозитория. Для совместимости с готовыми весами используйте Python 3.13 и версии зависимостей из проверки. DINOv2 при первом обращении загружает закреплённую версию модели. Скрипт подготовки набора скачивает исходные фото из Kaggle, проверяет архивы и формирует отложенную выборку.

```bash
python -m pip install -e '.[cv,dino,yolo]' \
  'scikit-learn==1.9.1' 'ultralytics==8.4.218' 'transformers==4.57.6'
python -m scripts.fetch_labeled_defects
python -m scripts.prepare_yolo_demo
```

```bash
python -m inspector run --data data/ml-demo-raw \
  --out runs/ml-yolo26n-background-v2-local \
  --artifact artifacts/ml-cube-v1-dinov2_hybrid \
  --locator artifacts/ml-yolo26n-background-v2/locator.json \
  --single-target --full-frame-artifact artifacts/ml-clean-full-dino-cv
python -m inspector serve --run runs/ml-yolo26n-background-v2-local \
  --artifact artifacts/ml-cube-v1-dinov2_hybrid \
  --locator artifacts/ml-yolo26n-background-v2/locator.json \
  --single-target --full-frame-artifact artifacts/ml-clean-full-dino-cv --port 8767
```

Снимки в `data/ml-demo-raw` взяты из проверочной выборки. В манифесте для инференса исходных рамок нет. Для пакетной перепроверки установите дополнение `.[vertex]` и добавьте к `run` флаги `--cloud-review gemini-4-argon --cloud-limit 6`. Для перепроверки загружаемых кадров добавьте `--cloud-review gemini-4-argon` к `serve`. Подключение использует активный ADC. Отдельное сравнение моделей на публичных фото запускается через `python -m scripts.benchmark_vertex`.

Для повторного обучения детектора на Apple Silicon скачайте исходный `yolo26n.pt` из [официального релиза Ultralytics](https://github.com/ultralytics/assets/releases) и поместите его в `artifacts/ml-yolo-assets/`. Списки фоновых кадров находятся в архиве весов. Подготовьте эти снимки командой `python -m scripts.fetch_public_manholes`. Манифест отдельной проверки кадров без люка включён в архив.

```bash
python -m scripts.prepare_yolo_demo --restore-distractors artifacts/ml-yolo-assets/distractors
python -m scripts.train_yolo_locator \
  --train data/ml-bench/verified/cube/train \
  --train-backgrounds artifacts/ml-yolo-negatives/train.txt \
  --val-backgrounds artifacts/ml-yolo-negatives/validation.txt \
  --holdout cube=data/ml-bench/verified/cube/holdout \
  --holdout deep=data/ml-bench/verified/deep/holdout \
  --distractors artifacts/ml-yolo-assets/distractors \
  --weights artifacts/ml-yolo-assets/yolo26n.pt \
  --out artifacts/ml-yolo26n-retrain \
  --imgsz 512 --epochs 22 --batch-size 16 --device mps
```
