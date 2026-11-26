# Воспроизведение модели шкафов v0.4

Из корня репозитория. Две выданные папки `Датасет-ДСС` и `Датасет-ДСС-2` должны находиться рядом с каталогом репозитория. Сохраните вложенные папки и таблицы XLSX. Фотографии и кэш признаков хранятся локально. В релизе доступны обезличенные прогнозы и сводные метрики.

## Подготовка

Проверенная среда использует Python 3.13 и timm 1.0.22. Первая команда подготовки скачивает публичные веса закреплённых ревизий и проверяет SHA-256. Для полного режима нужно около 2,7 ГБ весов. После подготовки обработка фотографий работает без сети.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cabinet,dev]'

export TORCH_HOME=artifacts/torch-cache
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
python -m scripts.cabinet_final prepare
export HF_HUB_OFFLINE=1

R=runs/cabinet-repro
A=artifacts/cabinet-repro
CACHE=data/final-audit/repro-control.npz
HR_DEVICE=mps
```

Для повторения замера HR используйте Apple MPS. При отсутствии MPS укажите `HR_DEVICE=cpu`. Значения отдельных оценок могут немного различаться из-за вычислений в FP32 на CPU.

## Обучение и проверка

Сначала строится контроль v0.2 для обоих дефектов. Команда сравнения сохраняет его оценки на пяти вариантах одного группового протокола. В каждом варианте внешние фолды разделяют визуальные группы, вложенные используются для выбора состава базового ансамбля.

```bash
python -m scripts.cabinet_final train \
  --source ../Датасет-ДСС --source ../Датасет-ДСС-2 \
  --reviewed-groups scripts/cabinet_groups.json \
  --out "$R/v02" --artifact "$A/v02"

python -m scripts.cabinet_compare --control-only \
  --source ../Датасет-ДСС --source ../Датасет-ДСС-2 \
  --reviewed-groups scripts/cabinet_groups.json \
  --out "$R/paired-control" --artifact "$A/paired-control" \
  --private-feature-cache "$CACHE"
```

Детектор формирует дополнительную область для головы типа шкафа. Если область отсутствует, тип остаётся результатом полноэкранного классификатора. Правило выбора действия использует исходную уверенность типа v0.2.

```bash
python -m scripts.benchmark_cabinet_roi \
  --source ../Датасет-ДСС --source ../Датасет-ДСС-2 \
  --reviewed-groups scripts/cabinet_groups.json \
  --detector artifacts/cabinet-roi-grounding/detector \
  --threads 3 --out "$R/crops"

python -m scripts.fit_cabinet_roi_type \
  --source ../Датасет-ДСС --source ../Датасет-ДСС-2 \
  --reviewed-groups scripts/cabinet_groups.json \
  --reference-features "$CACHE" \
  --roi-features "$R/crops/private-features.npz" \
  --roi-regions "$R/crops/private-regions.json" \
  --control "$R/paired-control/oof_comparison.json" \
  --control-model "$A/v02/model.joblib" \
  --out "$R/roi" --artifact "$A/roi"

python -m scripts.cabinet_final promote-roi \
  --control "$R/v02" --revision "$R/roi" \
  --model "$A/roi/model.joblib" \
  --out "$R/roi-release" --target "$A/roi-release/model.joblib"
```

Ветка HR обрабатывает четыре перекрывающихся ракурса кадра. В итоговый ансамбль входят средние оценки контрольной и HR-модели по каждому дефекту. Тип берётся из ROI-версии. Скрипт сборки сверяет фолды, разметку, исходные вероятности и пересчитывает очередь по итоговым оценкам.

```bash
python -m scripts.benchmark_cabinet_hr \
  --source ../Датасет-ДСС --source ../Датасет-ДСС-2 \
  --reviewed-groups scripts/cabinet_groups.json \
  --control-oof "$R/paired-control/oof_comparison.json" \
  --weights artifacts/cabinet-hr-siglip \
  --device "$HR_DEVICE" --threads 2 --out "$R/hr"

python -m scripts.fit_cabinet_hr_release \
  --reference-features "$CACHE" \
  --roi-release "$R/roi-release" --roi-paired "$R/roi" \
  --roi-model "$A/roi-release/model.joblib" \
  --control-oof "$R/paired-control/oof_comparison.json" \
  --hr "$R/hr" --out "$R/v04" --artifact "$A/v04"
```

Артефакт `$A/v04/model.joblib` содержит все обученные головы. В `$R/v04` находятся агрегаты, OOF-строки основного и дополнительных разбиений и очередь на все 62 снимка. В групповой оценке участвует 61 размеченный снимок. Повторные разбиения использовали тот же набор фотографий.

## Запуск

```bash
python -m scripts.cabinet_final predict /path/to/photo.jpg --model "$A/v04/model.joblib"
MGTS_CABINET_MODEL="$A/v04/model.joblib" \
  python -m uvicorn inspector.field:create_app --factory --host 127.0.0.1 --port 8883
```

Откройте [рабочее место мастера](http://127.0.0.1:8883/field). Если добавить `--activate` к команде сборки HR, она установит артефакт и соответствующие агрегаты в стандартные каталоги `artifacts/final-cabinet` и `runs/final-cabinet`. Тогда `--model` и `MGTS_CABINET_MODEL` при запуске можно опустить.

Для компактного режима используйте артефакт `$A/roi-release/model.joblib`. В свежей установке команда `python -m scripts.cabinet_final prepare --without-hr` скачает только его энкодеры. При запуске v0.4 без HR-весов API сообщает о переходе на v0.3. Источник типа отдельно указан для каждой карточки.
