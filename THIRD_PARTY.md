# Сторонние компоненты и данные

- Детектор дообучен с официальных весов Ultralytics YOLO26n. Опубликованные веса YOLO и код проекта распространяются на условиях AGPL‑3.0. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Условия Ultralytics](https://www.ultralytics.com/license)
- Локальный классификатор использует DINOv2 Small от Meta. Код загружает закреплённую версию исходной модели по необходимости. Обученные классификаторы проекта есть в релизе. [Модель DINOv2 Small](https://huggingface.co/facebook/dinov2-small)
- Для проверки использовались опубликованные подборки CubeAI, Deep Pratap и Manhole Covers Dataset. Скрипты обращаются к первоначальным источникам и проверяют хеши поддерживаемых архивов. [CubeAI](https://www.kaggle.com/datasets/cubeai/manhole-for-yolov8) · [Deep Pratap](https://www.kaggle.com/datasets/deeppratap/manhole-detection) · [Manhole Covers Dataset](https://huggingface.co/datasets/delima87/manhole_covers_dataset)
- Все фото, координаты и записи реестра в учебном демо синтетические. Gemini 4 Argon доступен как дополнительный сервис через настроенный пользователем Vertex AI.

## Финальная версия для РЩ и ТКШ

- В полном ансамбле используются Google SigLIP 2, Meta DINOv2 и Torchvision MobileNetV3. В репозитории опубликованы обученные линейные классификаторы. Предобученные кодировщики загружаются из официальных источников. [SigLIP 2](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli) · [DINOv2](https://huggingface.co/facebook/dinov2-small) · [MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html)
- Браузерный код использует INT8-конверсию DINOv2 от Xenova и Microsoft ONNX Runtime Web 1.22. Страница загружает их с собственного домена. Лицензии и уведомления включены в опубликованный каталог `predict/runtime/`. [Модель ONNX](https://huggingface.co/Xenova/dinov2-small) · [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- Анимация сцен выполнена в GSAP. Библиотека распространяется на условиях поставщика. Иллюстрации витрины созданы для демонстрации. [GSAP](https://gsap.com/standard-license/)
