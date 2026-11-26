# Сторонние компоненты и данные

- Детектор дообучен с официальных весов Ultralytics YOLO26n. Опубликованные веса YOLO и код проекта распространяются на условиях AGPL‑3.0. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Условия Ultralytics](https://www.ultralytics.com/license)
- Локальный классификатор использует DINOv2 Small от Meta. Код загружает закреплённую версию исходной модели по необходимости. Обученные классификаторы проекта есть в релизе. [Модель DINOv2 Small](https://huggingface.co/facebook/dinov2-small)
- Для проверки использовались опубликованные подборки CubeAI, Deep Pratap и Manhole Covers Dataset. Скрипты обращаются к первоначальным источникам и проверяют хеши поддерживаемых архивов. [CubeAI](https://www.kaggle.com/datasets/cubeai/manhole-for-yolov8) · [Deep Pratap](https://www.kaggle.com/datasets/deeppratap/manhole-detection) · [Manhole Covers Dataset](https://huggingface.co/datasets/delima87/manhole_covers_dataset)
- Все фото, координаты и записи реестра в учебном демо синтетические. Gemini 4 Argon доступен как дополнительный сервис через настроенный пользователем Vertex AI.

## Финальная версия для РЩ и ТКШ

- В полном ансамбле используются Google SigLIP 2, Meta DINOv2 и Torchvision MobileNetV3. В репозитории опубликованы обученные линейные классификаторы. Предобученные кодировщики загружаются из официальных источников. [SigLIP 2](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli) · [DINOv2](https://huggingface.co/facebook/dinov2-small) · [MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html)
- Браузерный код использует INT8-конверсию DINOv2 от Xenova и Microsoft ONNX Runtime Web 1.22. Страница загружает их с собственного домена. Лицензии и уведомления включены в опубликованный каталог `predict/runtime/`. [Модель ONNX](https://huggingface.co/Xenova/dinov2-small) · [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- Анимация сцен выполнена в GSAP. Библиотека распространяется на условиях поставщика. Иллюстрации витрины созданы для демонстрации. [GSAP](https://gsap.com/standard-license/)

- Для крупных фрагментов шкафов v0.4 использует So400m из семейства Google SigLIP 2 и Grounding DINO Tiny от IDEA Research для области шкафа. При подготовке команда проверяет закреплённые ревизии и SHA-256 весов. [SigLIP 2 So400m](https://huggingface.co/timm/vit_so400m_patch16_siglip_384.v2_webli) · [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)

## Материалы финальной продуктовой версии

Логотип МГТС и гарнитура MTS Text взяты из брендбука, предоставленного организаторами конкурса для оформления решения. Права на эти материалы принадлежат их правообладателям. Городская иллюстрация и маскот «Контур» созданы для этого прототипа. [Брендбук организаторов](https://disk.yandex.ru/d/uXw7Qs1IOmeFAg)

Второе мнение предоставляется облачным сервисом Google Vertex AI. В репозитории опубликованы клиент, зафиксированный промпт, сводные метрики и обезличенный пример ответа. [Описание подключения к Vertex AI](docs/FINAL.md)
