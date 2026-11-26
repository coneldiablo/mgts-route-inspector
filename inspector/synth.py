"""Synthetic route-inspection fixture for automated tests only.

The user-facing prototype is trained on public real photos (see
`scripts/fetch_public_manholes.py`). This generator stays in the repository to
keep fast, deterministic tests of the full chain: manifest -> quality gate ->
model -> unique cards -> UI. It is not a demo dataset and must not be presented
as field data.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SIZE = 512


@dataclass
class AssetPlan:
    asset_id: str
    seed: int
    style: str
    center: tuple[int, int]
    radius: tuple[int, int]
    states: list[str]
    conditions: list[str]
    criticality: float
    frames_per_visit: int = 1


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _asphalt(rng: np.random.Generator) -> Image.Image:
    # Grayscale base with a slight blue tint: one noise field, not per-channel rainbow.
    coarse = rng.normal(0, 16, (12, 12))
    coarse_img = Image.fromarray(np.uint8(np.clip(coarse + 128, 0, 255))).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    base = np.asarray(coarse_img, dtype=np.float32) - 128.0
    base = 78.0 + base * 0.75 + rng.normal(0, 6, (SIZE, SIZE))
    tint = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    arr = np.stack([base + tint[0], base + tint[1], base + tint[2]], axis=2)
    img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(2200):
        x, y = rng.integers(0, SIZE, 2)
        g = int(rng.integers(35, 150))
        d.point((int(x), int(y)), fill=(g, g, g))
    # cracks and patches vary the background enough to punish colour shortcuts
    for _ in range(int(rng.integers(2, 6))):
        points = [(int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))]
        for _ in range(int(rng.integers(2, 6))):
            points.append((points[-1][0] + int(rng.integers(-70, 70)), points[-1][1] + int(rng.integers(-70, 70))))
        d.line(points, fill=(int(rng.integers(25, 60)),) * 3, width=int(rng.integers(1, 4)))
    if rng.random() < 0.5:
        edge = int(rng.integers(0, 2))
        if edge == 0:
            d.rectangle([0, 0, SIZE, int(rng.integers(12, 55))], fill=(int(rng.integers(90, 150)),) * 3)
        else:
            d.rectangle([SIZE - int(rng.integers(15, 60)), 0, SIZE, SIZE], fill=(int(rng.integers(45, 80)),) * 3)
    return img


def _cover_pattern(
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    style: str,
    rng: np.random.Generator,
) -> None:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if style == "standard":
        base, ring, mark = (118, 120, 122), (82, 84, 87), (150, 151, 150)
    elif style == "old":
        base, ring, mark = (112, 91, 72), (66, 51, 41), (135, 112, 86)
    elif style == "dark_cover":
        base, ring, mark = (52, 53, 56), (31, 32, 35), (20, 20, 22)
    elif style == "stained":
        base, ring, mark = (104, 103, 100), (74, 73, 71), (58, 58, 57)
    else:  # dark_pattern: hard negative, central black emblem looks like a hole
        base, ring, mark = (64, 65, 68), (39, 40, 42), (18, 18, 20)
    d.ellipse(bbox, fill=base + (255,))
    for k in range(1, 4):
        inset_x, inset_y = int(rx * (0.18 * k)), int(ry * (0.18 * k))
        d.ellipse(
            (cx - rx + inset_x, cy - ry + inset_y, cx + rx - inset_x, cy + ry - inset_y),
            outline=ring + (230,),
            width=max(1, int(rng.integers(1, 4))),
        )
    for a in np.linspace(0, 2 * math.pi, int(rng.integers(10, 18)), endpoint=False):
        d.line(
            [(cx + int(rx * 0.22 * math.cos(a)), cy + int(ry * 0.22 * math.sin(a))),
             (cx + int(rx * 0.92 * math.cos(a)), cy + int(ry * 0.92 * math.sin(a)))],
            fill=ring + (120,),
            width=1,
        )
    if style == "dark_pattern":
        d.ellipse((cx - int(rx * 0.28), cy - int(ry * 0.28), cx + int(rx * 0.28), cy + int(ry * 0.28)), fill=(8, 8, 9, 255))
        d.ellipse((cx - int(rx * 0.20), cy - int(ry * 0.20), cx + int(rx * 0.20), cy + int(ry * 0.20)), fill=(24, 25, 28, 255))
    else:
        d.ellipse((cx - int(rx * 0.16), cy - int(ry * 0.16), cx + int(rx * 0.16), cy + int(ry * 0.16)), fill=mark + (255,))
        d.ellipse((cx - int(rx * 0.09), cy - int(ry * 0.09), cx + int(rx * 0.09), cy + int(ry * 0.09)), fill=ring + (255,))
    if style == "stained":
        for _ in range(5):
            px = cx + int(rng.integers(-int(rx * 0.55), int(rx * 0.55)))
            py = cy + int(rng.integers(-int(ry * 0.55), int(ry * 0.55)))
            rr = int(rng.integers(max(8, rx // 8), max(12, rx // 3)))
            d.ellipse((px - rr, py - int(rr * 0.8), px + rr, py + int(rr * 0.8)), fill=(42, 44, 45, 210))
    if style == "old":
        for _ in range(35):
            px, py = int(rng.integers(x1, x2)), int(rng.integers(y1, y2))
            if ((px - cx) / max(rx, 1)) ** 2 + ((py - cy) / max(ry, 1)) ** 2 < 0.85:
                d.ellipse((px, py, px + int(rng.integers(2, 9)), py + int(rng.integers(2, 9))), fill=(80, 54, 37, 190))
    img.paste(layer, (0, 0), layer.getchannel("A"))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.arc(bbox, 20, 200, fill=(0, 0, 0, 90), width=max(2, rx // 18))
    img.paste(overlay, (0, 0), overlay.getchannel("A"))


def _hatch(
    img: Image.Image,
    center: tuple[int, int],
    radius: tuple[int, int],
    state: str,
    style: str,
    rng: np.random.Generator,
    subtle: bool = False,
) -> tuple[int, int, int, int]:
    cx, cy = center
    rx, ry = radius
    rim_outer = (cx - rx, cy - ry, cx + rx, cy + ry)
    rim_inner = (cx - int(rx * 0.86), cy - int(ry * 0.86), cx + int(rx * 0.86), cy + int(ry * 0.86))
    d = ImageDraw.Draw(img)
    concrete = int(rng.integers(110, 160))
    d.ellipse(rim_outer, fill=(concrete, concrete - 3, concrete - 7), outline=(55, 55, 58), width=3)
    d.ellipse(rim_inner, fill=(40, 40, 42), outline=(35, 35, 37), width=2)
    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        bx = cx + int(rx * 0.92 * math.cos(a))
        by = cy + int(ry * 0.92 * math.sin(a))
        d.ellipse((bx - 3, by - 3, bx + 3, by + 3), fill=(70, 70, 72))

    cover_rx, cover_ry = int(rx * 0.82), int(ry * 0.82)
    if state == "normal":
        _cover_pattern(img, (cx - cover_rx, cy - cover_ry, cx + cover_rx, cy + cover_ry), style, rng)
        return rim_outer
    # Open hole: dark depth gradient, with the cover leaning nearby or absent.
    hole_box = (cx - cover_rx, cy - cover_ry, cx + cover_rx, cy + cover_ry)
    d.ellipse(hole_box, fill=(8, 8, 9))
    for k in range(6, 0, -1):
        frac = k / 6.0
        b = (cx - int(cover_rx * frac), cy - int(cover_ry * frac), cx + int(cover_rx * frac), cy + int(cover_ry * frac))
        shade = int(12 + 26 * (1 - frac))
        d.ellipse(b, fill=(shade, shade, shade + 1))
    if state == "open" and rng.random() < 0.75:
        offset = int(rx * rng.uniform(0.95, 1.35))
        lean = (cx + offset - cover_rx // 2, cy + int(ry * rng.uniform(0.05, 0.3)) - cover_ry // 2,
                cx + offset + cover_rx // 2, cy + int(ry * rng.uniform(0.05, 0.3)) + cover_ry // 2)
        if lean[2] < SIZE and lean[0] > 0:
            _cover_pattern(img, lean, style, rng)
    if state == "ajar":
        shift = int(cover_rx * (rng.uniform(0.06, 0.15) if subtle else rng.uniform(0.18, 0.38)))
        _cover_pattern(img, (cx - cover_rx + shift, cy - cover_ry, cx + cover_rx + shift, cy + cover_ry), style, rng)
    return rim_outer


def _occlude(img: Image.Image, rng: np.random.Generator) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    if rng.random() < 0.5:
        # A car body slicing across the lower part of the frame.
        top = int(rng.integers(330, 430))
        d.rounded_rectangle([-30, top, SIZE + 30, SIZE + 30], radius=24, fill=(int(rng.integers(25, 90)), int(rng.integers(30, 90)), int(rng.integers(40, 110)), 255))
        d.rounded_rectangle([-10, top + 30, 200, top + 90], radius=12, fill=(80, 95, 115, 220))
        d.rounded_rectangle([230, top + 25, 470, top + 95], radius=14, fill=(70, 85, 105, 220))
    else:
        for _ in range(int(rng.integers(2, 5))):
            x, y = int(rng.integers(50, 460)), int(rng.integers(330, 500))
            d.ellipse((x, y, x + int(rng.integers(45, 130)), y + int(rng.integers(18, 45))), fill=(int(rng.integers(70, 120)), int(rng.integers(90, 140)), int(rng.integers(40, 80)), 235))


def _weather(img: Image.Image, condition: str, rng: np.random.Generator) -> Image.Image:
    if condition == "wet":
        img = ImageEnhance.Brightness(img).enhance(0.78)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for _ in range(8):
            x, y = int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))
            d.ellipse((x, y, x + int(rng.integers(20, 90)), y + int(rng.integers(8, 25))), fill=(150, 180, 210, 35))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    elif condition == "shadow":
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.polygon([(0, 0), (SIZE, 0), (SIZE, int(SIZE * 0.4)), (0, int(SIZE * 0.65))], fill=(0, 0, 0, 100))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    elif condition == "night":
        img = ImageEnhance.Brightness(img).enhance(0.38)
        arr = np.asarray(img, dtype=np.float32)
        arr[:, :, 2] += 12
        arr += rng.normal(0, 5, arr.shape)
        img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    elif condition == "snow":
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for _ in range(16):
            x, y = int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))
            d.ellipse((x, y, x + int(rng.integers(20, 90)), y + int(rng.integers(10, 40))), fill=(235, 238, 240, 165))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    elif condition == "occluded":
        _occlude(img, rng)
    elif condition == "occluded_hard":
        d = ImageDraw.Draw(img, "RGBA")
        top = int(rng.integers(120, 210))
        d.rounded_rectangle([-30, top, SIZE + 30, SIZE + 30], radius=30, fill=(35, 42, 55, 255))
        d.rounded_rectangle([-15, top + 22, 210, top + 88], radius=12, fill=(70, 84, 105, 230))
        d.rounded_rectangle([240, top + 18, 490, top + 92], radius=14, fill=(60, 76, 98, 230))
        d.rectangle([0, SIZE - 70, SIZE, SIZE], fill=(22, 24, 28, 255))
    elif condition == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=5.2))
    elif condition == "dark":
        img = ImageEnhance.Brightness(img).enhance(0.16)
    return img


def _plan_assets(n_assets: int, seed: int) -> list[AssetPlan]:
    rng = _rng(seed)
    plans: list[AssetPlan] = []
    styles = ["standard", "old", "dark_pattern", "dark_cover", "stained"]
    states_cycle = [
        ["normal", "open"],
        ["normal", "normal"],
        ["open", "open"],
        ["normal", "ajar"],
        ["open", "normal"],
        ["normal", "open"],
        ["normal", "normal"],
        ["ajar", "open"],
    ]
    condition_sets = [
        ["day", "day"],
        ["day", "night"],
        ["wet", "day"],
        ["day", "shadow"],
        ["night", "snow"],
        ["day", "occluded"],
        ["day", "occluded_hard"],
    ]
    for i in range(n_assets):
        style = styles[i % len(styles)]
        states = list(states_cycle[i % len(states_cycle)])
        # A few hard cases: closed cover with a dark painted emblem, barely ajar cover.
        if i % 11 == 3:
            style = "dark_pattern"
        elif i % 9 == 2:
            style = "stained"
        conditions = list(condition_sets[(i // 2) % len(condition_sets)])
        if i % 13 == 5:
            conditions[1] = "hard_negative"
        if i % 17 == 7:
            conditions[1] = "hard_positive"
            states[1] = "ajar"
        if i % 19 == 2:
            conditions[1] = "subtle"
            states[1] = "ajar"
        plans.append(
            AssetPlan(
                asset_id=f"KKS-{i + 1:04d}",
                seed=seed * 1000 + i,
                style=style,
                center=(int(rng.integers(190, 325)), int(rng.integers(175, 300))),
                radius=(int(rng.integers(78, 112)), 0),
                states=states,
                conditions=conditions,
                criticality=round(float(rng.uniform(0.3, 0.9)), 2),
                frames_per_visit=3 if i % 12 == 2 else 1,
            )
        )
    for p in plans:
        # Keep perspective consistent per object but not perfectly round.
        rr = _rng(p.seed)
        p.radius = (p.radius[0], int(p.radius[0] * rr.uniform(0.58, 0.92)))
    return plans


def generate(out_dir: str | Path, n_assets: int = 48, seed: int = 7) -> dict:
    out = Path(out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plans = _plan_assets(n_assets, seed)
    rows: list[dict] = []
    for plan in plans:
        for visit, (state, condition) in enumerate(zip(plan.states, plan.conditions)):
            for frame in range(plan.frames_per_visit):
                rng = _rng(plan.seed + visit * 31 + frame * 7)
                img = _asphalt(rng)
                jitter = (int(rng.integers(-14, 15)), int(rng.integers(-12, 13))) if frame else (0, 0)
                center = (plan.center[0] + jitter[0], plan.center[1] + jitter[1])
                if condition == "hard_negative":
                    state_eff = "normal"
                    cond_eff = "hard_negative"
                elif condition == "hard_positive":
                    state_eff = "ajar"
                    cond_eff = "hard_positive"
                else:
                    state_eff = state
                    cond_eff = condition
                label = "open_hatch" if state_eff in {"open", "ajar"} else "normal"
                bbox = _hatch(
                    img, center, plan.radius, state_eff, plan.style, rng,
                    subtle=cond_eff in {"hard_positive", "subtle"},
                )
                img = _weather(img, cond_eff, rng)
                # A small camera overlay: timestamp only, no asset ID that could leak the split.
                d = ImageDraw.Draw(img)
                d.rectangle([0, SIZE - 18, SIZE, SIZE], fill=(20, 22, 25))
                d.text((8, SIZE - 16), f"MGTScam  {2026 + visit:04d}-11-{10 + visit:02d}", fill=(190, 195, 200))
                if frame == 0:
                    d.text((8, 8), "MGTScam", fill=(220, 220, 220))
                img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
                quality = 72 if condition not in {"night", "dark"} else 55
                image_id = f"{plan.asset_id}__v{visit + 1}"
                if frame:
                    image_id += f"_f{frame + 1}"
                if condition == "blur":
                    image_id += "_blur"
                path = images_dir / f"{image_id}.jpg"
                img.save(path, "JPEG", quality=quality)
                x1, y1, x2, y2 = bbox
                rows.append(
                    {
                        "image": f"images/{path.name}",
                        "image_id": image_id,
                        "asset_id": plan.asset_id,
                        "object_type": "well",
                        "captured_at": f"2026-11-{10 + visit:02d}T{9 + (plan.seed % 8):02d}:{(plan.seed * 7 + visit * 13 + frame) % 60:02d}:00+03:00",
                        "lat": round(55.751 + (int(plan.asset_id[-4:]) % 40) * 0.0009, 6),
                        "lon": round(37.610 + (int(plan.asset_id[-4:]) // 40) * 0.0011 + visit * 0.00002, 6),
                        "label": label,
                        "condition": cond_eff,
                        "bbox_x1": max(0, x1 - 12),
                        "bbox_y1": max(0, y1 - 12),
                        "bbox_x2": min(SIZE, x2 + 12),
                        "bbox_y2": min(SIZE, y2 + 12),
                        "criticality": plan.criticality,
                        "source": "synthetic",
                    }
                )

    # A few unusable frames from normal objects: blur/dark must become "request another frame",
    # not a false finding and not a silent miss of a known defect.
    rng = _rng(seed + 999)
    normal_plans = [p for p in plans if p.states[-1] == "normal"][:6]
    for idx, plan in enumerate(normal_plans):
        condition = "blur" if idx % 2 == 0 else "dark"
        rng2 = _rng(plan.seed + 700)
        img = _asphalt(rng2)
        bbox = _hatch(img, plan.center, plan.radius, "normal", plan.style, rng2)
        img = _weather(img, condition, rng2)
        image_id = f"{plan.asset_id}__unusable_{idx + 1}"
        path = images_dir / f"{image_id}.jpg"
        img.save(path, "JPEG", quality=50)
        x1, y1, x2, y2 = bbox
        rows.append(
            {
                "image": f"images/{path.name}",
                "image_id": image_id,
                "asset_id": plan.asset_id,
                "object_type": "well",
                "captured_at": "2026-11-12T22:10:00+03:00",
                "lat": round(55.751 + (int(plan.asset_id[-4:]) % 40) * 0.0009, 6),
                "lon": round(37.610 + (int(plan.asset_id[-4:]) // 40) * 0.0011, 6),
                "label": "unusable",
                "condition": condition,
                "bbox_x1": max(0, x1 - 12),
                "bbox_y1": max(0, y1 - 12),
                "bbox_x2": min(SIZE, x2 + 12),
                "bbox_y2": min(SIZE, y2 + 12),
                "criticality": plan.criticality,
                "source": "synthetic",
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    readme = f"""# Синтетический демонстрационный набор

{n_assets} объектов, {len(df)} кадров. Это **не данные МГТС** и не доказательство
качества на реальных снимках. Набор нужен, чтобы проверить весь путь:
manifest → проверка кадра → модель → одна карточка на объект → интерфейс мастера.

- `label = open_hatch` — видимый открытый проём (включая слегка сдвинутую крышку)
- `label = normal` — крышка на месте
- `label = unusable` — кадр непригоден (сильное размытие/темнота)
- `condition` — день, ночь, влага, тень, снег, перекрытие, сложный негатив

Сгенерировано `python -m inspector synth --out data/demo --assets {n_assets} --seed {seed}`.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    _contact_sheet(out, df)
    return {"n_assets": n_assets, "n_images": int(len(df)), "out": str(out)}


def _contact_sheet(out: Path, df) -> None:
    sample = df.sample(n=min(20, len(df)), random_state=3)
    tile = 192
    cols = 5
    rows = math.ceil(len(sample) / cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 22)), (24, 26, 30))
    d = ImageDraw.Draw(sheet)
    for i, (_, row) in enumerate(sample.iterrows()):
        img = Image.open(out / row["image"]).convert("RGB").resize((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 22)
        sheet.paste(img, (x, y))
        label = str(row["label"]).replace("open_hatch", "open")
        d.text((x + 5, y + tile + 4), f"{row['asset_id']} · {label} · {row['condition']}", fill=(220, 220, 220))
    sheet.save(out / "contact_sheet.jpg", quality=88)
