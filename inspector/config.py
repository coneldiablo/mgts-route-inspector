"""Configuration and domain vocabulary for the route-inspection prototype.

Everything here is a hypothesis that must be checked against the final MGTS
brief. The values are deliberately explicit so they can be changed in one place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QualityConfig:
    """When is a frame good enough for a visual conclusion?"""

    min_width: int = 480
    min_height: int = 360
    min_sharpness: float = 28.0  # variance of Laplacian, computed on a 512px copy
    min_brightness: float = 20.0
    max_brightness: float = 232.0
    min_contrast: float = 6.0
    min_area_fraction: float = 0.012
    uncertain_band: float = 0.07

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_config(profile: str = "default") -> QualityConfig:
    """Named quality profiles.

    `default` is for full-resolution route photos. `lowres` is for the public
    224x224 web dataset used in the prototype: its frames are small and the
    open-hatch class is often soft, so full-resolution thresholds would reject
    almost everything. The profile is persisted with the model and reused by
    `run`/`serve`, so training and inference apply the same gate.
    """
    if profile == "default":
        return QualityConfig()
    if profile == "lowres":
        return QualityConfig(
            min_width=150,
            min_height=110,
            min_sharpness=2.0,
            min_contrast=2.5,
        )
    raise ValueError(f"Неизвестный профиль качества: {profile}")


@dataclass
class ModelConfig:
    backbone: str = "auto"
    normal_label: str = "normal"
    unusable_label: str = "unusable"
    threshold_mode: str = "f1"  # f1 | budget
    max_fpr: float = 0.10
    cv_folds: int = 5
    random_state: int = 42
    image_size: int = 224
    quality_profile: str = "default"
    comparison_backbone: str | None = "classical"
    defect_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PUBLIC_SOURCE = "public:delima87/manhole_covers_dataset"
PUBLIC_REVIEW_LABEL = "unconfirmed_cover"
PUBLIC_TASK = "Подтверждение видимости крышки на фото (manhole / void)"


DEFAULT_SIGN_ACTIONS: dict[str, dict[str, Any]] = {
    PUBLIC_REVIEW_LABEL: {
        "title": "Модель не распознала крышку на снимке",
        "short": "Проверить видимость крышки",
        "severity": 0.5,
        "action": "Проверить, попал ли объект в кадр. Если крышку трудно различить, запросить новый снимок.",
    },
    "open_hatch": {
        "title": "Видимый открытый проём на месте крышки",
        "short": "Открытый проём",
        "severity": 1.0,
        "action": (
            "Проверить колодец: на месте штатной крышки виден открытый проём. "
            "Сверить с нарядом и плановыми работами; при подтверждении направить "
            "бригаду для осмотра и ограждения."
        ),
    },
    "cover_shifted": {
        "title": "Смещённая крышка колодца",
        "short": "Смещённая крышка",
        "severity": 0.8,
        "action": (
            "Проверить посадку крышки и отсутствие открытого проёма. "
            "Сверить с плановыми работами, при необходимости направить бригаду."
        ),
    },
    "damaged_body": {
        "title": "Видимое повреждение наружного элемента",
        "short": "Повреждение",
        "severity": 0.8,
        "action": (
            "Осмотреть повреждённый элемент с более близкого ракурса и решить, "
            "требуется ли ремонт по регламенту."
        ),
    },
}


@dataclass
class AppConfig:
    quality: QualityConfig = field(default_factory=QualityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sign_actions: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SIGN_ACTIONS.items()}
    )
    include_normal_cards: bool = False
    synthetic_warning: bool = True

    def resolve_sign(self, sign: str | None) -> dict[str, Any]:
        if not sign or sign in {"normal", "unusable"}:
            return {
                "title": "Признак из проверяемого списка не выявлен",
                "short": "Без замечания",
                "severity": 0.0,
                "action": "Действий по этому признаку не требуется.",
            }
        if sign in self.sign_actions:
            return self.sign_actions[sign]
        return {
            "title": f"Наблюдаемый признак: {sign}",
            "short": sign,
            "severity": 0.5,
            "action": "Проверить объект и подтвердить признак по регламенту.",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality.to_dict(),
            "model": self.model.to_dict(),
            "sign_actions": self.sign_actions,
            "include_normal_cards": self.include_normal_cards,
            "synthetic_warning": self.synthetic_warning,
        }
