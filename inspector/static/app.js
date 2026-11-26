const state = {
  cards: [],
  summary: null,
  eval: null,
  filter: "queue",
  locationFilter: "all",
  query: "",
  threshold: 0.5,
  selectedId: null,
  preview: null,
  uploaded: null,
  pendingLocations: new Set(),
  reviewDrafts: new Map(),
};

const STATUS_ORDER = { finding: 0, uncertain: 1, rephoto: 2, normal: 3 };
const STATUS_LABEL = {
  finding: "Требует проверки",
  uncertain: "Слабый сигнал",
  rephoto: "Нужен другой кадр",
  normal: "Без замечания",
};
const REVIEW_LABEL = {
  new: "Не проверено",
  confirmed: "Подтверждено",
  rejected: "Отклонено",
  rephoto_requested: "Запрошен кадр",
};
const LOCATION_LABEL = {
  confirmed: "Место подтверждено",
  reported: "Привязка из ведомости",
  candidate: "Нужна сверка",
  ambiguous: "Нужна сверка",
  unlocated: "Без местоположения",
};
const LOCATION_SOURCE = {
  registry: "Реестр объектов",
  asset_id: "ID из ведомости",
  object_coordinates: "Координаты объекта из ведомости",
  camera_coordinates: "Координаты камеры",
  manual: "Сверено мастером",
};
const LOCATION_REASON = {
  registry_coordinates_conflict: "Точка из ведомости расходится с координатами в реестре.",
  registry_type_conflict: "Тип объекта в ведомости отличается от типа в реестре.",
  asset_id_not_in_registry: "ID из ведомости отсутствует в загруженном реестре.",
  registry_required: "Для поиска кандидатов по точке съёмки нужен реестр.",
  no_nearby_asset: "В заданном радиусе не нашлось подходящих объектов реестра.",
};
const OBJECT_LABEL = {
  well: "Кабельный колодец",
  manhole: "Люк",
  pole: "Опора",
  cabinet: "Шкаф",
};
const MODEL_LABEL = {
  mobilenet_v3_small: "MobileNetV3-Small",
  hybrid_cnn_cv: "MobileNetV3 + признаки CV",
  classical_cv: "Классификатор CV",
  dinov2_small: "DINOv2 Small",
  dinov2_hybrid: "DINOv2 + признаки CV",
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pct(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function fmtDate(value) {
  if (!value) return "время не передано";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function mediaUrl(path) {
  return path ? `/media/${path}` : "";
}

function objectLabel(type) {
  return Object.hasOwn(OBJECT_LABEL, type) ? OBJECT_LABEL[type] : type || "Объект";
}

function point(lat, lon) {
  if (lat == null || lon == null || String(lat).trim() === "" || String(lon).trim() === "") return null;
  const y = Number(lat);
  const x = Number(lon);
  return Number.isFinite(y) && Math.abs(y) <= 90 && Number.isFinite(x) && Math.abs(x) <= 180
    ? { lat: y, lon: x } : null;
}

function locationOf(card) {
  if (Object.hasOwn(LOCATION_LABEL, card.location?.status)) return card.location;
  const coords = point(card.lat, card.lon);
  return coords
    ? { status: "reported", source: "object_coordinates" }
    : { status: "unlocated", source: "none" };
}

function objectPoint(card, location = locationOf(card)) {
  return (location.status === "confirmed" || location.status === "reported") ? point(card.lat, card.lon) : null;
}

function coordsText(coords) {
  return `${coords.lat.toFixed(6)}, ${coords.lon.toFixed(6)}`;
}

function needsLocationReview(location) {
  return location.status === "candidate" || location.status === "ambiguous";
}

function matchesLocationFilter(card) {
  const location = locationOf(card);
  if (state.locationFilter === "located") return !!objectPoint(card, location);
  if (state.locationFilter === "pending") return needsLocationReview(location);
  if (state.locationFilter === "confirmed" || state.locationFilter === "unlocated") {
    return location.status === state.locationFilter;
  }
  return true;
}

function statusOf(card, threshold) {
  if (card.localization?.source === "full_frame" || threshold === Number(card.threshold)) return card.status;
  if (card.kind === "rephoto" || !card.quality || !card.quality.ok) return "rephoto";
  const p = Number(card.p_finding || 0);
  if (p >= threshold) return "finding";
  if (state.summary?.model?.mode === "defect_type") return "uncertain";
  if (p >= Math.max(0, threshold - (state.summary?.quality_config?.uncertain_band ?? 0.07))) return "uncertain";
  return "normal";
}

function computeCard(card, threshold) {
  if (card.localization?.source === "full_frame" || threshold === Number(card.threshold)) {
    return { ...card, ui_status: card.status, ui_short: card.finding_short, ui_title: card.finding_title,
      ui_action: card.action, ui_priority: card.priority };
  }
  const status = statusOf(card, threshold);
  const p = Number(card.p_finding || 0);
  const severity = Number(card.candidate_severity ?? 0.5);
  const reasons = (card.quality && card.quality.reasons) || [];
  let short = card.candidate_short || "Без замечания";
  let title = card.candidate_title || "Признак не выявлен";
  let action = card.candidate_action || "Действий по проверяемому признаку не требуется.";
  let priority = p * severity + 0.1 * Number(card.criticality ?? 0.5);
  if (status === "rephoto") {
    short = "Нужен другой кадр";
    title = "Недостаточно данных для оценки";
    action = reasons.length
      ? `Запросить повторный кадр: ${reasons.join(", ")}.`
      : "Запросить повторный кадр: качество снимка недостаточно для вывода.";
    priority = 0.35;
  } else if (status === "uncertain") {
    action = state.summary?.model?.mode === "defect_type"
      ? "Нужно вручную уточнить вид повреждения."
      : "Сигнал слабее порога. Проверьте вручную или запросите кадр крупнее.";
    priority *= 0.7;
  } else if (status === "normal") {
    const publicCover = card.candidate_sign === "unconfirmed_cover";
    short = publicCover ? "Крышка в кадре" : "Без замечания";
    title = publicCover ? "Модель распознала крышку на снимке" : "Признак из проверяемого списка не выявлен";
    action = publicCover ? "Кадр можно передать мастеру для оценки состояния крышки." : "Действий по признаку не требуется.";
    priority = 0;
  }
  return { ...card, ui_status: status, ui_short: short, ui_title: title, ui_action: action, ui_priority: priority };
}

function filteredCards() {
  const t = state.threshold;
  let cards = state.cards.map((c) => computeCard(c, t));
  if (state.filter !== "all") {
    cards = cards.filter((c) => (state.filter === "queue" ? c.ui_status !== "normal" : c.ui_status === state.filter));
  }
  cards = cards.filter(matchesLocationFilter);
  const q = state.query.trim().toLowerCase();
  if (q) {
    cards = cards.filter((c) =>
      [c.asset_id, c.image_id, c.candidate_short, c.finding_short, c.coordinates,
        c.location?.matched_asset_id, c.location?.address, ...(c.location?.candidates ?? []).map((p) => p.asset_id)]
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }
  cards.sort((a, b) => {
    const d = STATUS_ORDER[a.ui_status] - STATUS_ORDER[b.ui_status];
    if (d !== 0) return d;
    if (b.ui_priority !== a.ui_priority) return b.ui_priority - a.ui_priority;
    return String(a.asset_id).localeCompare(String(b.asset_id));
  });
  return cards;
}

function countsAtThreshold() {
  const counts = { finding: 0, uncertain: 0, rephoto: 0, normal: 0 };
  state.cards.forEach((card) => {
    counts[statusOf(card, state.threshold)] += 1;
  });
  counts.queue = counts.finding + counts.uncertain + counts.rephoto;
  counts.all = state.cards.length;
  return counts;
}

function renderKpis() {
  const s = state.summary || {};
  const e = state.eval || {};
  const counts = countsAtThreshold();
  const coverage = Number(s.coverage_decisive_share || 0);
  const model = s.model || {};
  const modelName = MODEL_LABEL[model.backbone] || model.backbone || "—";
  const recall = e.recall != null ? pct(e.recall) : "—";
  const precision = e.precision != null ? pct(e.precision) : "—";
  const hasFrameIdCoverage = s.n_assets_with_provided_id != null;
  const assetHint = hasFrameIdCoverage
    ? `${s.n_processed ?? "—"} кадров · ${s.n_frames_without_provided_id ?? 0} без ID`
    : `${s.n_processed ?? "—"} кадров`;
  const kpis = [
    { label: "В очереди", value: counts.queue, hint: "карточек для мастера", cls: "alert" },
    { label: "Требуют проверки", value: counts.finding, hint: "выше порога", cls: "alert" },
    { label: "Нужен кадр", value: counts.rephoto, hint: "качество недостаточно", cls: "info" },
    { label: hasFrameIdCoverage ? "Объекты с ID" : "Объектов", value: s.n_assets_with_provided_id ?? s.n_assets ?? "—", hint: assetHint, cls: "" },
    { label: "Покрытие", value: pct(coverage), hint: "вывод возможен", cls: "good" },
    e.mode === "defect_type"
      ? { label: "Macro-F1 по видам", value: e.type_macro_f1 != null ? pct(e.type_macro_f1) : "—",
          hint: `${e.evaluation} · вывод по ${e.n_accepted} из ${e.n_images} фото`, cls: "" }
      : { label: "Recall / Precision", value: recall, hint: `precision ${precision} · ${s.holdout_eval ? "отложенные фото" : "OOF"} · порог ${e.threshold ?? "—"}`, cls: "" },
  ];
  $("#kpis").innerHTML = kpis.map((k) => `
    <div class="kpi ${k.cls}">
      <div class="kpi-label">${esc(k.label)}</div>
      <div class="kpi-value">${esc(k.value)}</div>
      <div class="kpi-hint">${esc(k.hint)}</div>
    </div>`).join("");

  $("#modelChip").textContent = [s.locator?.backend, modelName].filter(Boolean).join(" + ");
  const uploadCloud = $("#uploadCloudNotice");
  uploadCloud.hidden = !s.upload_cloud_review;
  uploadCloud.textContent = s.upload_cloud_review
    ? `Если локальная модель не нашла рамку или сомневается в состоянии люка, фото или его фрагмент отправится в ${s.upload_cloud_review}.`
    : "";
  $("#syntheticNotice").hidden = !s.synthetic;
  const sources = Object.keys(s.source_counts || {});
  const publicNotice = $("#publicNotice");
  publicNotice.hidden = !sources.some((src) => src.startsWith("public:"));
  publicNotice.textContent = sources.some((src) => src.startsWith("public:verified/"))
    ? "Публичные проверочные фото. Модель различает открытый проём и закрытый люк."
    : "Публичное демо проверяет, видна ли крышка на фото. Открытые люки требуют отдельной разметки.";
  $("#thresholdValue").textContent = Number(state.threshold).toFixed(2);

  const countsEls = {
    countQueue: counts.queue,
    countFinding: counts.finding,
    countUncertain: counts.uncertain,
    countRephoto: counts.rephoto,
    countAll: counts.all,
  };
  Object.entries(countsEls).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
  $("#queueMeta").textContent = `${counts.queue} в очереди · ${counts.all} карточек`;
}

function renderQueue(cards) {
  const list = $("#queueList");
  if (!cards.length) {
    list.innerHTML = `<div class="empty-state" style="min-height:220px"><p>По этому фильтру объектов нет.</p></div>`;
    return;
  }
  list.innerHTML = cards.map((card) => {
    const image = mediaUrl(card.image_file || card.crop_file);
    const p = Number(card.p_finding || 0);
    const selected = state.selectedId === card.card_id && !state.preview;
    const duplicates = Number(card.duplicates_merged || 0);
    return `
      <button type="button" class="queue-item ${selected ? "selected" : ""}" data-card-id="${esc(card.card_id)}" aria-current="${selected ? "true" : "false"}">
        <span class="queue-thumb">
          ${image ? `<img src="${esc(image)}" alt="" loading="lazy">` : ""}
        </span>
        <span class="queue-content">
          <span class="queue-title">
            <span class="asset-id"><span class="status-dot dot-${esc(card.ui_status)}"></span>${esc(card.location?.matched_asset_id || card.asset_id)}</span>
            <span class="p-score">${esc(card.ui_status === "rephoto" ? "нет вывода" : p.toFixed(2))}</span>
          </span>
          <span class="queue-sub">${esc(card.ui_short)}</span>
          <span class="queue-foot">
            <span>${esc(fmtDate(card.captured_at))}</span>
            ${duplicates ? `<span class="dup">Повторы · ${duplicates}</span>` : ""}
            ${needsLocationReview(locationOf(card)) ? `<span class="map-review-hint">Сверить место</span>` : ""}
            ${card.review_status && card.review_status !== "new" ? `<span>${esc(REVIEW_LABEL[card.review_status] || card.review_status)}</span>` : ""}
          </span>
        </span>
      </button>`;
  }).join("");
}

function renderLocationMap(cards) {
  const map = $("#locationMap");
  const points = cards.flatMap((card) => {
    const coords = objectPoint(card);
    return coords ? [{ ...coords, card }] : [];
  });
  const pending = cards.filter((card) => needsLocationReview(locationOf(card))).length;
  $("#locationMeta").textContent = `${points.length} с координатами${pending ? ` · ${pending} к сверке` : ""}`;
  if (!points.length) {
    map.innerHTML = `<p class="map-empty">${pending
      ? "Подтвердите объект в карточке, чтобы добавить его на схему."
      : "По текущим фильтрам нет карточек с координатами объекта."}</p>`;
    return;
  }
  const bounds = points.reduce((b, p) => ({
    north: Math.max(b.north, p.lat), south: Math.min(b.south, p.lat),
    east: Math.max(b.east, p.lon), west: Math.min(b.west, p.lon),
  }), { north: -90, south: 90, east: -180, west: 180 });
  const centerLat = (bounds.north + bounds.south) / 2;
  const centerLon = (bounds.east + bounds.west) / 2;
  const lngScale = Math.max(0.01, Math.cos(centerLat * Math.PI / 180));
  const mapAspect = 2.5;
  const ySpan = Math.max(bounds.north - bounds.south, (bounds.east - bounds.west) * lngScale / mapAspect, 0.00015) / 0.76;
  const xSpan = ySpan * mapAspect;
  const groups = new Map();
  points.forEach(({ lat, lon, card }) => {
    const x = 50 + 100 * (lon - centerLon) * lngScale / xSpan;
    const y = 50 + 100 * (centerLat - lat) / ySpan;
    const key = `${Math.round(x / 8)}:${Math.round(y / 16)}`;
    const group = groups.get(key);
    if (!group) groups.set(key, { x, y, cards: [card] });
    else group.cards.push(card);
  });
  const markers = [...groups.values()].map(({ x, y, cards: nearby }) => {
    const card = nearby.find((c) => c.card_id === state.selectedId) || nearby[0];
    const selected = card.card_id === state.selectedId;
    const id = locationOf(card).matched_asset_id || card.asset_id;
    const groupLabel = nearby.length > 1 ? `, карточек рядом: ${nearby.length}` : "";
    return `<button type="button" class="map-marker map-marker-${esc(locationOf(card).status)}${selected ? " is-selected" : ""}"
      data-map-card="${esc(card.card_id)}" aria-label="${esc(`Открыть ${id}${groupLabel}`)}"
      aria-current="${selected ? "true" : "false"}" title="${esc(id + groupLabel)}" style="--x:${x.toFixed(2)}%;--y:${y.toFixed(2)}%">
      ${nearby.length > 1 ? nearby.length : '<span aria-hidden="true"></span>'}
    </button>`;
  }).join("");
  map.innerHTML = `
    <div class="map-field" role="group" aria-label="Координаты объектов, север сверху, без картографической подложки">
      <span class="map-north" aria-hidden="true">↑ С</span>
      ${markers}
    </div>
    <div class="map-legend">
      <span><i class="map-key map-key-confirmed"></i>Подтверждено</span>
      <span><i class="map-key"></i>Из ведомости</span>
      <span class="map-caption">Без подложки</span>
    </div>`;
}

function bboxOverlay(card) {
  if (!card.bbox_pct) return "";
  const [x1, y1, x2, y2] = card.bbox_pct;
  const label = card.localization?.source === "vertex_confirmed" ? "рамка Gemini" : "область объекта";
  return `<div class="bbox" style="left:${x1}%;top:${y1}%;width:${x2 - x1}%;height:${y2 - y1}%">
      <span class="bbox-label">${label}</span>
    </div>`;
}

function badge(status, text) {
  return `<span class="badge badge-${esc(status)}"><span class="status-dot dot-${esc(status)}"></span>${esc(text)}</span>`;
}

function renderLocation(card, preview) {
  const location = locationOf(card);
  const coords = objectPoint(card, location);
  const camera = point(location.camera_lat, location.camera_lon);
  const source = LOCATION_SOURCE[location.source];
  const pending = needsLocationReview(location);
  const manual = location.source === "manual" && !preview;
  const candidates = pending || manual
    ? (location.candidates || []).filter((item) => item.asset_id !== location.matched_asset_id)
    : [];
  const disabled = state.pendingLocations.has(card.card_id) ? "disabled" : "";
  const id = location.matched_asset_id || (location.status === "reported" && location.source === "asset_id" ? card.asset_id : null);
  const prompt = (!manual && LOCATION_REASON[location.reason]) || (pending
    ? "Сопоставьте снимок с записью реестра и подтвердите нужный объект."
    : !coords && location.status === "reported"
      ? "Координаты объекта не переданы."
      : !coords ? "Добавьте ID из реестра или координаты объекта." : "");
  const candidateList = candidates.length ? `
    <ul class="location-candidates" aria-label="Кандидаты из реестра">
      ${candidates.map((candidate) => {
        const distance = Number(candidate.distance_m);
        const distanceLabel = candidate.distance_m != null && Number.isFinite(distance)
          ? `${Math.round(distance).toLocaleString("ru-RU")} м` : "";
        const candidateCoords = point(candidate.lat, candidate.lon);
        const meta = [candidate.address, candidateCoords ? coordsText(candidateCoords) : "", distanceLabel].filter(Boolean);
        return `<li class="location-candidate">
          <div><strong>${esc(candidate.asset_id)}</strong>${candidate.object_type ? `<span class="candidate-type"> · ${esc(objectLabel(candidate.object_type))}</span>` : ""}
          ${meta.length ? `<span class="candidate-meta">${esc(meta.join(" · "))}</span>` : ""}</div>
          ${preview ? "" : `<button type="button" class="btn btn-location" data-location-asset="${esc(candidate.asset_id)}"
            aria-label="${esc(`Привязать к объекту ${candidate.asset_id}`)}" ${disabled}>Привязать</button>`}
        </li>`;
      }).join("")}
    </ul>` : "";
  return `
    <section class="panel location-panel" aria-labelledby="locationTitle">
      <div class="location-head">
        <h3 id="locationTitle" tabindex="-1">Местоположение объекта</h3>
        <span class="location-state location-state-${esc(location.status)}">${esc(LOCATION_LABEL[location.status])}</span>
      </div>
      ${source || id || coords || camera ? `
        <dl class="location-facts">
          ${id ? `<div><dt>ID объекта</dt><dd>${esc(id)}</dd></div>` : ""}
          ${source ? `<div><dt>Источник привязки</dt><dd>${esc(source)}</dd></div>` : ""}
          ${location.address ? `<div><dt>Адрес</dt><dd>${esc(location.address)}</dd></div>` : ""}
          ${coords ? `<div><dt>Координаты объекта</dt><dd><span class="location-coords">${coordsText(coords)}</span>
            <button class="btn btn-ghost copy-coords" type="button" data-copy-coords="${coordsText(coords)}" aria-label="Скопировать координаты объекта">Копировать</button></dd></div>` : ""}
          ${camera ? `<div><dt>Точка съёмки</dt><dd class="location-coords">${coordsText(camera)}</dd></div>` : ""}
        </dl>` : ""}
      ${prompt ? `<p class="muted location-prompt">${esc(prompt)}</p>` : ""}
      ${manual && candidateList ? `
        <details class="location-alternatives"><summary>Изменить привязку</summary>${candidateList}</details>` : candidateList}
    </section>`;
}

function renderReview(card, preview) {
  if (preview) {
    return `<div class="review"><div class="review-head"><h3>Новый кадр</h3>
      <span class="review-state">не записан в очередь прогона</span></div>
      <p class="muted">Этот результат показывает путь «файл → проверка качества → признак → карточка». Для очереди используйте пакетный прогон по папке.</p></div>`;
  }
  if (state.threshold !== Number(card.threshold)) {
    return `<p class="muted">Для записи решения верните порог прогона ${esc(card.threshold)}.</p>`;
  }
  const current = card.review_status && card.review_status !== "new" ? card.review_status : "new";
  return `
    <div class="review">
      <div class="review-head">
        <h3>Решение мастера</h3>
        <span class="review-state">Статус: <b>${esc(REVIEW_LABEL[current] || current)}</b></span>
      </div>
      <textarea class="review-note" id="reviewNote" aria-label="Комментарий мастера" placeholder="Комментарий: что проверить, какой ракурс нужен, что сказал наряд">${esc(state.reviewDrafts.get(card.card_id) ?? card.review_note ?? "")}</textarea>
      <div class="review-actions">
        <button class="btn btn-confirm" data-decision="confirmed">Подтвердить замечание</button>
        <button class="btn btn-rephoto" data-decision="rephoto_requested">Запросить кадр</button>
        <button class="btn btn-reject" data-decision="rejected">Отклонить</button>
      </div>
      ${card.review_updated_at ? `<div class="review-history">Обновлено: ${esc(fmtDate(card.review_updated_at))}</div>` : ""}
    </div>`;
}

function renderCloudReview(cloud) {
  if (!cloud) return "";
  const labels = { normal: "крышка на месте", open_hatch: "проём открыт", uncertain: "состояние неясно" };
  const messages = {
    confirmed_bbox: "Рамку предложил Gemini, состояние совпало с локальной оценкой",
    local_disagreement: "Модели разошлись, оставлена локальная оценка",
    advisory: "Дополнительная оценка",
    unavailable: "Сервис не вернул оценку",
  };
  const message = messages[cloud.confirmation] || messages.advisory;
  const status = cloud.confirmation === "unavailable" ? message : `${message} · ${labels[cloud.prediction] || labels.uncertain}`;
  return `
    <div class="panel" style="margin-bottom:12px">
      <h3>Перепроверка ${esc(cloud.model)}</h3>
      <p>${esc(status)}</p>
      ${cloud.evidence ? `<p class="muted">${esc(cloud.evidence)}</p>` : ""}
    </div>`;
}

function renderFrames(card) {
  const frames = (card.frames || []).filter((f) => f.image_file);
  if (frames.length <= 1) return "";
  return `
    <div class="panel" style="margin-bottom:12px">
      <h3>Кадры объекта (${frames.length}) · повторы не создают новую задачу</h3>
      <div class="frames">
        ${frames.map((f) => `
          <a class="frame ${f.image_id === card.image_id ? "best" : ""}" href="${esc(mediaUrl(f.image_file))}" target="_blank" rel="noopener" title="${esc(f.image_id)} · ${esc(f.status_label || f.status)}">
            <img src="${esc(mediaUrl(f.image_file))}" alt="${esc(f.image_id)}" loading="lazy">
            <span>${esc(fmtDate(f.captured_at))} <em>${esc(Number(f.p_finding || 0).toFixed(2))}</em></span>
          </a>`).join("")}
      </div>
    </div>`;
}

function renderDetail() {
  const detail = $("#detail");
  const card = state.preview || state.cards.find((c) => c.card_id === state.selectedId);
  if (!card) {
    detail.innerHTML = `<div class="empty-state">
      <div class="empty-icon">◎</div>
      <h2>Выберите объект в очереди</h2>
      <p>К карточке приложены исходный кадр, геопривязка и предлагаемое действие.</p>
    </div>`;
    return;
  }
  const c = computeCard(card, state.preview ? Number(card.threshold) : state.threshold);
  const p = Number(c.p_finding || 0);
  const q = c.quality || {};
  const review = c.review_status && c.review_status !== "new" ? c.review_status : "new";
  const preview = !!state.preview;
  const image = mediaUrl(c.image_file);
  const crop = mediaUrl(c.crop_file);
  const title = c.ui_title;
  const location = locationOf(c);
  const fieldPoint = objectPoint(c, location);
  const headerMeta = [objectLabel(c.object_type), fmtDate(c.captured_at), fieldPoint ? coordsText(fieldPoint) : null].filter(Boolean);
  const backbone = state.summary?.model?.backbone;
  const modelName = MODEL_LABEL[backbone] || backbone || "—";
  const detectorName = state.summary?.locator?.backend;
  detail.innerHTML = `
    <div class="detail-head">
      <div class="detail-title">
        <h2 tabindex="-1">${esc(location.matched_asset_id || c.asset_id)}</h2>
        <p>${esc(headerMeta.join(" · "))}</p>
      </div>
      <div class="detail-badges">
        ${badge(c.ui_status, STATUS_LABEL[c.ui_status])}
        ${review !== "new" ? `<span class="badge badge-neutral">${esc(REVIEW_LABEL[review] || review)}</span>` : ""}
        ${preview ? `<span class="badge badge-neutral">новый кадр</span>` : ""}
      </div>
    </div>

    <div class="media-row">
      <div class="image-stage">
        <div class="image-frame">
          ${image ? `<img src="${esc(image)}" alt="Исходный кадр ${esc(c.image_id)}">` : `<span class="muted">нет изображения</span>`}
          ${bboxOverlay(c)}
        </div>
      </div>
      <div class="evidence-sidebar">
        ${renderLocation(c, preview)}
        ${crop ? `<div class="crop-stage">
          <img src="${esc(crop)}" alt="Фрагмент объекта">
          <span>Фрагмент для модели</span>
        </div>` : ""}
      </div>
    </div>

    <div class="info-grid">
      <div class="panel">
        <h3>Наблюдаемый признак</h3>
        <div class="big">${esc(c.ui_short)}</div>
        <p class="muted">${esc(title)}</p>
        <div class="reason">${esc(c.why || "")}</div>
      </div>
      <div class="panel">
        <h3>Основание</h3>
        <div class="score-row">
          <div class="score-bar"><div class="score-fill" style="width:${Math.round(p * 100)}%"></div></div>
          <span class="score-value">${q.ok ? p.toFixed(2) : "нет вывода"}</span>
        </div>
        <div class="metric-list">
          <div class="metric-line"><span>порог решения</span><b>${Number(preview ? c.threshold : state.threshold).toFixed(2)}</b></div>
          <div class="metric-line"><span>качество кадра</span><b>${q.ok ? "пригоден" : "недостаточно"}</b></div>
          <div class="metric-line"><span>резкость / яркость</span><b>${esc(q.sharpness ?? "—")} / ${esc(q.brightness ?? "—")}</b></div>
          <div class="metric-line"><span>модель</span><b>${esc(modelName)}</b></div>
          ${detectorName ? `<div class="metric-line"><span>детектор</span><b>${esc(detectorName)}</b></div>` : ""}
          <div class="metric-line"><span>кадров объекта</span><b>${esc(c.frames_count ?? 1)}</b></div>
        </div>
        <p class="muted" style="font-size:10.5px;margin-top:7px">Оценка модели — не вероятность аварии.</p>
      </div>
    </div>

    <div class="panel action-panel" style="margin-bottom:12px">
      <h3>Предлагаемое действие</h3>
      <p>${esc(c.ui_action)}</p>
      ${Number(c.duplicates_merged || 0) > 0 ? `<p class="muted">Повторные кадры объединены: +${esc(c.duplicates_merged)} к той же карточке, отдельная задача не создаётся.</p>` : ""}
      ${!q.ok && q.reasons?.length ? `<p class="muted">Причины отказа: ${esc(q.reasons.join(", "))}.</p>` : ""}
    </div>

    ${renderCloudReview(c.cloud_review)}
    ${renderFrames(c)}
    ${renderReview(c, preview)}
  `;
}

function renderAll() {
  const visible = filteredCards();
  if (!state.preview) {
    if (!visible.some((c) => c.card_id === state.selectedId)) {
      state.selectedId = visible[0]?.card_id ?? null;
    }
  }
  renderKpis();
  renderQueue(visible);
  renderLocationMap(visible);
  renderDetail();
}

function selectCard(cardId, { focusQueue = false, jumpToDetail = false } = {}) {
  const changed = state.selectedId !== cardId || state.preview;
  state.preview = null;
  state.selectedId = cardId;
  const visible = filteredCards();
  renderQueue(visible);
  renderLocationMap(visible);
  renderDetail();
  const detail = $("#detail");
  if (changed) detail.scrollTop = 0;
  const el = $(`.queue-item[data-card-id="${CSS.escape(cardId)}"]`);
  if (jumpToDetail) {
    detail.scrollIntoView({ block: "start" });
    $(".detail-title h2", detail)?.focus({ preventScroll: true });
  } else if (el) {
    el.scrollIntoView({ block: "nearest" });
    if (focusQueue) el.focus({ preventScroll: true });
  }
}

async function submitDecision(decision) {
  const card = state.cards.find((c) => c.card_id === state.selectedId);
  if (!card) return;
  const note = $("#reviewNote")?.value || "";
  try {
    const res = await fetch(`/api/cards/${encodeURIComponent(card.card_id)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, note }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    card.review_status = data.review.decision;
    card.review_note = data.review.note;
    card.review_updated_at = data.review.updated_at;
    if (state.reviewDrafts.get(card.card_id) === note) state.reviewDrafts.delete(card.card_id);
    toast(`Сохранено: ${REVIEW_LABEL[decision] || decision}`);
    renderAll();
  } catch (err) {
    toast(`Не удалось сохранить: ${err.message}`, true);
  }
}

async function submitLocation(assetId, button) {
  const card = state.cards.find((entry) => entry.card_id === state.selectedId);
  if (!card || state.preview || state.pendingLocations.has(card.card_id)) return;
  state.pendingLocations.add(card.card_id);
  $$("[data-location-asset]", $("#detail")).forEach((el) => { el.disabled = true; });
  button.textContent = "Сохраняю…";
  try {
    const res = await fetch(`/api/cards/${encodeURIComponent(card.card_id)}/location`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId }),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok || !data?.card?.location) {
      throw new Error(typeof data?.detail === "string" ? data.detail : `Ошибка сервера (${res.status})`);
    }
    const { location, lat, lon, coordinates } = data.card;
    Object.assign(card, { location, lat, lon, coordinates });
    if (state.selectedId === card.card_id && !matchesLocationFilter(card)) {
      state.locationFilter = "all";
      $("#locationFilter").value = "all";
    }
    renderAll();
    if (state.selectedId === card.card_id) $("#locationTitle")?.focus({ preventScroll: true });
    toast(`Место подтверждено · ${location.matched_asset_id || assetId}`);
  } catch (err) {
    toast(`Привязка не сохранена · ${err.message}`, true);
  } finally {
    state.pendingLocations.delete(card.card_id);
    if (state.selectedId === card.card_id) {
      $$("[data-location-asset]", $("#detail")).forEach((el) => { el.disabled = false; });
      if (button.isConnected) {
        button.textContent = "Привязать";
        button.focus({ preventScroll: true });
      }
    }
  }
}

async function copyCoordinates(coords) {
  try {
    await navigator.clipboard.writeText(coords);
    toast("Координаты скопированы");
  } catch {
    toast("Выделите координаты в карточке и скопируйте их", true);
  }
}

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast ${isError ? "error" : ""}`;
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 3200);
}

function uploadResultHtml(card) {
  const c = computeCard(card, Number(card.threshold));
  return `
    <div class="upload-card">
      <img src="${esc(mediaUrl(c.image_file))}" alt="Проверяемый кадр">
      <div>
        <h3>${esc(c.asset_id)} · ${esc(c.ui_short)}</h3>
        <p>${esc(c.ui_status === "rephoto" ? c.ui_action : `Модель: ${Number(c.p_finding || 0).toFixed(2)}. ${c.why || ""}`)}</p>
        <div class="row">
          ${badge(c.ui_status, STATUS_LABEL[c.ui_status])}
          <button class="btn btn-ghost" type="button" id="openUploaded">Открыть в панели</button>
        </div>
      </div>
    </div>
    ${renderCloudReview(c.cloud_review)}`;
}

async function analyzeUpload(event) {
  event.preventDefault();
  const file = $("#fileInput").files[0];
  if (!file) { toast("Выберите файл", true); return; }
  const btn = $("#analyzeBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Проверяю…`;
  const form = new FormData();
  form.append("file", file);
  form.append("asset_id", $("#assetInput").value || "UPLOAD-001");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.uploaded = data.card;
    $("#uploadResult").innerHTML = uploadResultHtml(data.card);
    $("#openUploaded").addEventListener("click", () => {
      state.preview = state.uploaded;
      state.selectedId = null;
      closeUpload();
      renderAll();
    });
  } catch (err) {
    $("#uploadResult").innerHTML = `<p class="muted">Ошибка: ${esc(err.message)}</p>`;
    toast(`Не удалось проверить кадр: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Проверить кадр";
  }
}

function openUpload() { $("#uploadModal").hidden = false; }
function closeUpload() { $("#uploadModal").hidden = true; }

function bindEvents() {
  $("#filters").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-filter]");
    if (!btn) return;
    state.filter = btn.dataset.filter;
    $$(".filter").forEach((el) => {
      el.classList.toggle("active", el === btn);
      el.setAttribute("aria-pressed", String(el === btn));
    });
    renderAll();
  });

  $("#searchInput").addEventListener("input", (event) => {
    state.query = event.target.value;
    renderAll();
  });

  $("#locationFilter").addEventListener("change", (event) => {
    state.locationFilter = event.target.value;
    renderAll();
  });

  $("#resetThreshold").addEventListener("click", () => {
    if (!state.summary) return;
    state.threshold = Number(state.summary.threshold);
    $("#thresholdInput").value = state.threshold;
    renderAll();
  });
  $("#thresholdInput").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    renderAll();
  });

  const selectFromWorkspace = (cardId) => selectCard(cardId, {
    focusQueue: true,
    jumpToDetail: matchMedia("(max-width: 960px)").matches,
  });
  $("#queueList").addEventListener("click", (event) => {
    const item = event.target.closest("[data-card-id]");
    if (item) selectFromWorkspace(item.dataset.cardId);
  });

  $("#locationMap").addEventListener("click", (event) => {
    const marker = event.target.closest("[data-map-card]");
    if (marker) selectFromWorkspace(marker.dataset.mapCard);
  });

  $("#detail").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-decision]");
    if (btn) submitDecision(btn.dataset.decision);
    const candidate = event.target.closest("[data-location-asset]");
    if (candidate) submitLocation(candidate.dataset.locationAsset, candidate);
    const copy = event.target.closest("[data-copy-coords]");
    if (copy) copyCoordinates(copy.dataset.copyCoords);
  });
  $("#detail").addEventListener("input", (event) => {
    if (event.target.id === "reviewNote" && state.selectedId) {
      state.reviewDrafts.set(state.selectedId, event.target.value);
    }
  });

  $("#uploadBtn").addEventListener("click", openUpload);
  $("#closeUpload").addEventListener("click", closeUpload);
  $("#uploadModal").addEventListener("click", (event) => {
    if (event.target === $("#uploadModal")) closeUpload();
  });
  $("#uploadForm").addEventListener("submit", analyzeUpload);
  $("#fileInput").addEventListener("change", () => {
    const file = $("#fileInput").files[0];
    $("#dropText").textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} МБ` : "Выберите файл или перетащите снимок";
  });
  const drop = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => {
    event.preventDefault(); drop.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => {
    event.preventDefault(); drop.classList.remove("drag");
  }));
  drop.addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length) {
      $("#fileInput").files = event.dataTransfer.files;
      $("#fileInput").dispatchEvent(new Event("change"));
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!$("#uploadModal").hidden) {
        closeUpload();
        $("#uploadBtn").focus();
      } else {
        const thresholdPopup = $(".threshold-menu");
        if (thresholdPopup.open) {
          thresholdPopup.open = false;
          $("summary", thresholdPopup).focus();
        }
      }
    }
    if (!(event.target === document.body || event.target.closest?.(".queue-item"))) return;
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const cards = filteredCards();
    if (!cards.length) return;
    const idx = cards.findIndex((c) => c.card_id === state.selectedId);
    const next = event.key === "ArrowDown" ? Math.min(cards.length - 1, idx + 1) : Math.max(0, idx - 1);
    event.preventDefault();
    selectCard(cards[next].card_id, { focusQueue: true });
  });
}

async function init() {
  bindEvents();
  try {
    const [summary, cards] = await Promise.all([
      fetch("/api/summary").then((r) => { if (!r.ok) throw new Error("summary"); return r.json(); }),
      fetch("/api/cards").then((r) => { if (!r.ok) throw new Error("cards"); return r.json(); }),
    ]);
    state.summary = summary;
    state.eval = summary.holdout_eval || summary.eval || null;
    state.cards = cards;
    const t = Number(summary.threshold ?? 0.5);
    state.threshold = t;
    $("#thresholdInput").value = state.threshold;
    const first = filteredCards()[0];
    if (first) state.selectedId = first.card_id;
    renderAll();
  } catch (err) {
    $("#detail").innerHTML = `<div class="empty-state"><h2>Не удалось загрузить прогон</h2><p>${esc(err.message)}. Проверьте, что выполнен <code>python -m inspector run</code> и сервер запущен из нужной папки.</p></div>`;
    toast("Ошибка загрузки данных прогона", true);
  }
}

init();
