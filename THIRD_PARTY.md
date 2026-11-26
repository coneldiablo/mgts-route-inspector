# Сторонние компоненты и данные

- Детектор дообучен с официальных весов Ultralytics YOLO26n. Опубликованные веса YOLO и код проекта распространяются на условиях AGPL‑3.0. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Условия Ultralytics](https://www.ultralytics.com/license)
- Локальный классификатор использует DINOv2 Small от Meta. Код загружает закреплённую версию исходной модели по необходимости. Обученные классификаторы проекта есть в релизе. [Модель DINOv2 Small](https://huggingface.co/facebook/dinov2-small)
- Для проверки использовались опубликованные подборки CubeAI, Deep Pratap и Manhole Covers Dataset. Скрипты обращаются к первоначальным источникам и проверяют хеши поддерживаемых архивов. [CubeAI](https://www.kaggle.com/datasets/cubeai/manhole-for-yolov8) · [Deep Pratap](https://www.kaggle.com/datasets/deeppratap/manhole-detection) · [Manhole Covers Dataset](https://huggingface.co/datasets/delima87/manhole_covers_dataset)
- Все фото, координаты и записи реестра в учебном демо синтетические. Gemini 4 Argon доступен как дополнительный сервис через настроенный пользователем Vertex AI.
