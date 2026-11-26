const get = id => document.getElementById(id);
const controls = Object.fromEntries(['query', 'object', 'condition', 'action', 'sort'].map(key => [key, get(key)]));
const conditionLabels = {normal:'Норма', damage:'Признаки повреждения', insufficient:'Недостаточно данных'};
const actionLabels = {repair:'Кандидат на ремонт', inspect:'Проверить', no_action:'Без действий'};
const signalLabels = {mechanical:'МП · механическое повреждение', coating:'ЛКП · дефект покрытия'};
const basisLabels = {
  repair:'Модель обнаружила выраженный визуальный признак. Мастер проверяет тип объекта и подтверждает объём работ.',
  inspect:'Мастеру нужно сверить признаки и тип объекта по исходному кадру. Оценки МП и ЛКП приведены отдельно.',
  no_action:'По оценкам модели видимые признаки механического повреждения и дефекта покрытия не выявлены.',
  quality:'Качества кадра недостаточно, чтобы оценить состояние. Нужна пересъёмка или ручная проверка.',
};
const priority = {repair:0, inspect:1, no_action:2};
const coordinates = 'Не подтверждены';
const pageSize = 12;
const percent = new Intl.NumberFormat('ru-RU', {style:'percent', maximumFractionDigits:1});
const collator = new Intl.Collator('ru-RU', {numeric:true, sensitivity:'base'});
let dataset = [];
let filtered = [];
let page = 0;
let opener;
let argonConfig;

function element(name, cls, text) {
  const node = document.createElement(name);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function arrow() {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  const path = document.createElementNS(ns, 'path');
  svg.setAttribute('viewBox', '0 0 20 20');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('fill', 'none');
  path.setAttribute('d', 'M7 4l6 6-6 6');
  svg.append(path);
  return svg;
}

function tag(category, labels) {
  return element('span', `tag ${category}`, labels[category]);
}

function searchValue(value) {
  return value.toLocaleUpperCase('ru-RU').replace(/Ё/g, 'Е').replace(/[\s–—-]/g, '').replace(/K/g, 'К');
}

function compare(a, b, sort) {
  const tie = () => collator.compare(a.id, b.id);
  if (sort === 'id') return tie();
  if (sort === 'mp' || sort === 'paint') {
    const signal = sort === 'mp' ? 'mechanical' : 'coating';
    return b.fitted.scores[signal] - a.fitted.scores[signal] || tie();
  }
  return priority[a.fitted.action] - priority[b.fitted.action]
    || Math.max(...Object.values(b.fitted.scores)) - Math.max(...Object.values(a.fitted.scores)) || tie();
}

function opinion(local, argon) {
  if (!argonConfig || !argon) return null;
  const {weight, threshold} = argonConfig;
  return Object.fromEntries(Object.keys(signalLabels).map(code => {
    const base = local.scores[code];
    const cloud = argon.status === 'ok' && argon.readable ? argon.scores[code] : null;
    return [code, {base, cloud, blend:cloud === null ? base : (1 - weight) * base + weight * cloud,
      disagreement:cloud !== null && (base >= threshold) !== (cloud >= threshold)}];
  }));
}

function hasDisagreement(result) {
  const scores = opinion(result.fitted, result.argon);
  return Boolean(scores && Object.values(scores).some(signal => signal.disagreement));
}

function createRow(result) {
  const li = element('li');
  const button = element('button', 'row');
  button.type = 'button';
  button.setAttribute('aria-haspopup', 'dialog');
  const identity = element('span', 'identity');
  identity.append(element('b', '', result.id), element('small', '', result.fitted.object));
  if (hasDisagreement(result)) identity.append(element('span', 'discord', 'Расхождение Qwen'));
  button.append(identity, tag(result.fitted.condition, conditionLabels), tag(result.fitted.action, actionLabels),
    element('span', 'coord', `Координаты ${coordinates.toLowerCase()}`), arrow());
  button.addEventListener('click', () => openCard(result, button));
  li.append(button);
  return li;
}

function renderRows() {
  const start = page * pageSize;
  const visible = filtered.slice(start, start + pageSize);
  get('rows').replaceChildren(...visible.map(createRow));
  get('result-count').textContent = `Найдено ${filtered.length} из ${dataset.length} кадров`;
  get('export').disabled = !filtered.length;
  get('table').hidden = !filtered.length;
  get('empty').hidden = Boolean(filtered.length);
  get('empty').textContent = 'По этим фильтрам ничего не найдено. Измените условия или сбросьте фильтры.';
  get('pages').hidden = filtered.length <= pageSize;
  get('page-position').textContent = `${start + 1}–${start + visible.length} из ${filtered.length}`;
  get('prev').disabled = page === 0;
  get('next').disabled = start + pageSize >= filtered.length;
}

function applyFilters() {
  const query = searchValue(controls.query.value.trim());
  const object = controls.object.value;
  const condition = controls.condition.value;
  const action = controls.action.value;
  const sort = controls.sort.value;
  const disagreement = get('disagreement').checked;
  filtered = dataset.filter(result =>
    (!query || searchValue(`${result.id} ${result.fitted.object}`).includes(query))
    && (!object || object === result.fitted.object)
    && (!condition || condition === result.fitted.condition)
    && (!action || action === result.fitted.action)
    && (!disagreement || hasDisagreement(result)));
  filtered.sort((a, b) => compare(a, b, sort));
  document.querySelectorAll('.summary-card').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.action === action));
  });
  page = 0;
  renderRows();
}

function fact(label, value) {
  const item = element('div');
  const content = element('dd');
  if (typeof value === 'string') content.textContent = value;
  else content.append(value);
  item.append(element('dt', '', label), content);
  return item;
}

function detailPrediction(prediction) {
  const data = element('dl', 'detail-data');
  data.append(fact('Состояние', tag(prediction.condition, conditionLabels)),
    fact('Рекомендация', tag(prediction.action, actionLabels)),
    fact('Тип по модели', `${prediction.object} · ${percent.format(prediction.objectScore)}`),
    fact('Пригодность кадра', prediction.qualityOk ? 'Проверку прошёл' : 'Нужна пересъёмка'));
  const signalGroup = element('div', 'detail-signals');
  for (const [code, label] of Object.entries(signalLabels)) {
    const signal = element('div');
    const title = element('strong', '', label);
    const score = element('strong');
    score.append(element('span', '', percent.format(prediction.scores[code])));
    const description = !prediction.qualityOk ? 'Нужен пригодный кадр'
      : prediction.findings.includes(code) ? 'Признак отмечен моделью' : 'Признак не отмечен';
    signal.append(title, score, element('small', '', description));
    signalGroup.append(signal);
  }
  return [data, element('p', 'detail-basis', basisLabels[prediction.basis]), signalGroup];
}

function argonOpinion(local, argon, primary) {
  const scores = opinion(local, argon);
  if (!scores) return [];
  const box = element('details', 'detail-oof detail-argon');
  box.open = primary;
  box.append(element('summary', '', 'Второе мнение Qwen3.8-27B'));
  const body = element('div', 'oof-body');
  body.append(element('p', '', 'Сохранённый ответ по этому снимку. Тип и рекомендацию определяет локальная модель.'));
  const {weight, threshold} = argonConfig;
  const info = argon.status === 'timeout' ? 'Таймаут Qwen. Использованы локальные оценки.'
    : argon.readable ? `${percent.format(1-weight)} локальная оценка + ${percent.format(weight)} Qwen. Расхождение отмечено при разных решениях на пороге ${percent.format(threshold)}.`
      : 'Qwen не смог оценить читаемость. Использованы локальные оценки.';
  body.append(element('p', 'argon-note', info));
  const values = element('div', 'detail-signals');
  for (const [code, valuesForSignal] of Object.entries(scores)) {
    const {base, cloud, blend, disagreement} = valuesForSignal;
    const cell = element('div');
    cell.append(element('strong', '', signalLabels[code]),
      element('small', '', `Локально ${percent.format(base)}`),
      element('small', '', cloud === null ? 'Qwen —' : `Qwen ${percent.format(cloud)}`),
      element('strong', '', `${cloud === null ? 'Итог' : 'Смесь'} ${percent.format(blend)}`));
    if (disagreement) cell.append(element('span', 'discord', 'Расхождение'));
    values.append(cell);
  }
  body.append(values);
  if (argon.cues?.length) {
    body.append(element('p', 'argon-note', `Признаки по Qwen — ${argon.cues.join(' · ')}.`));
  }
  box.append(body);
  return [box];
}

function openCard(result, button) {
  opener = button;
  get('card-title').textContent = `${result.id} · ${result.fitted.object}`;
  const base = element('p', 'detail-version', 'Прогноз после обучения');
  const location = element('p', 'detail-basis', `Координаты объекта ${coordinates.toLowerCase()}.`);
  const oof = element('details', 'detail-oof');
  oof.append(element('summary', '', 'Проверочный прогноз (OOF)'));
  const comparison = element('div', 'oof-body');
  if (result.oof) {
    comparison.append(element('p', '', 'Прогноз модели, обученной без группы этого кадра.'), ...detailPrediction(result.oof), ...argonOpinion(result.oof, result.argon, false));
  } else {
    comparison.append(element('p', '', 'Для кадра нет разметки. Он обработан после обучения и не участвовал в расчёте метрик.'));
  }
  oof.append(comparison);
  get('card-body').replaceChildren(base, ...detailPrediction(result.fitted), ...argonOpinion(result.fitted, result.argon, true), location, oof);
  get('card').showModal();
}

function csvField(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function downloadCsv() {
  const header = ['Кадр', 'Тип объекта', 'Состояние', 'Рекомендация', 'МП, %', 'ЛКП, %', 'Координаты', 'Разметка'];
  if (argonConfig) header.push('Ответ Qwen', 'Qwen МП, %', 'Qwen ЛКП, %', 'Смесь МП, %', 'Смесь ЛКП, %', 'Расхождения');
  const csvPercent = value => value === null ? '' : (100 * value).toFixed(2).replace('.', ',');
  const body = filtered.map(row => [row.id, row.fitted.object, conditionLabels[row.fitted.condition],
    actionLabels[row.fitted.action], csvPercent(row.fitted.scores.mechanical), csvPercent(row.fitted.scores.coating), coordinates,
    row.labeled ? 'Есть' : 'Нет']);
  if (argonConfig) filtered.forEach((row, index) => {
    const scores = opinion(row.fitted, row.argon);
    const split = Object.keys(signalLabels).filter(code => scores[code].disagreement).map(code => code === 'mechanical' ? 'МП' : 'ЛКП');
    const status = row.argon.status === 'timeout' ? 'Таймаут' : row.argon.readable ? 'Сохранён' : 'Кадр нечитаем';
    body[index].push(status, csvPercent(scores.mechanical.cloud), csvPercent(scores.coating.cloud),
      csvPercent(scores.mechanical.blend), csvPercent(scores.coating.blend), split.join(', '));
  });
  const csv = '\ufeff' + [header, ...body].map(fields => fields.map(csvField).join(';')).join('\r\n') + '\r\n';
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const link = element('a');
  link.href = url;
  link.download = 'mgts-registry.csv';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function resultsSummary(validation) {
  const labeled = dataset.filter(result => result.labeled).length;
  get('scope').textContent = `${dataset.length} кадров обработано`;
  get('labels').textContent = `${labeled} с разметкой · ${dataset.length - labeled} без разметки`;
  get('count-all').textContent = dataset.length;
  for (const action of Object.keys(actionLabels)) {
    get(`count-${action}`).textContent = dataset.filter(row => row.fitted.action === action).length;
  }
  get('summary').hidden = false;
  get('intro-meta').hidden = false;
  get('validation-facts').append(
    element('p', '', `Проверка OOF охватывает ${labeled} кадр с разметкой, ${validation.groups} группы близких кадров и ${validation.folds} фолдов. В каждом фолде модель обучали без целых групп из проверочной части.`),
    element('p', '', `Macro-F1 по МП и ЛКП ${percent.format(validation.defectMacroF1)}. Точность определения типа ${percent.format(validation.objectAccuracy)}. Повреждение размечено у ${validation.repairConfirmed} из ${validation.repairTotal} проверочных кандидатов на ремонт.`),
    element('p', '', 'Основной список показывает прогнозы после обучения. Метрики рассчитываются по проверочным прогнозам OOF, которые открываются отдельно в карточке.'),
  );
  const argon = argonConfig?.validation;
  if (argon) {
    get('validation-facts').append(element('p', '', `С сохранёнными ответами Qwen и переходом на локальную оценку при таймауте macro-F1 составил ${percent.format(argon.macroF1)} на тех же ${labeled} кадрах OOF. Определены ${argon.tp} из ${argon.positives} кадров с повреждениями. Ложные срабатывания — ${argon.fp} из ${argon.negatives} штатных кадров.`));
  }
  get('validation').hidden = false;
}

get('filters').addEventListener('submit', event => event.preventDefault());
get('disagreement').addEventListener('change', applyFilters);
for (const [name, control] of Object.entries(controls)) control.addEventListener(name === 'query' ? 'input' : 'change', applyFilters);
get('clear-filters').addEventListener('click', () => { get('filters').reset(); applyFilters(); });
get('export').addEventListener('click', downloadCsv);
for (const [id, delta] of [['prev', -1], ['next', 1]]) {
  get(id).addEventListener('click', () => { page += delta; renderRows(); get('records').focus(); });
}
document.querySelectorAll('.summary-card').forEach(button => button.addEventListener('click', () => {
  controls.action.value = button.dataset.action;
  applyFilters();
}));
get('card').addEventListener('close', () => opener?.focus());

try {
  const response = await fetch(new URL('./results.json', import.meta.url));
  if (!response.ok) throw Error('Registry unavailable');
  const result = await response.json();
  if (result.version !== 1 || !Array.isArray(result.rows)) throw Error('Invalid registry');
  dataset = result.rows;
  argonConfig = result.argonConfig;
  get('disagreement-wrap').hidden = !argonConfig;
  resultsSummary(result.validation);
  applyFilters();
} catch {
  get('result-count').textContent = 'Результаты временно недоступны.';
  get('empty').hidden = false;
  get('empty').textContent = 'Обновите страницу для повторной загрузки реестра.';
}
