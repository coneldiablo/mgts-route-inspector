import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import metrics from '../runs/final-cabinet/public_metrics.json'
import vertex from '../runs/cabinet-vertex/public-benchmark.json'
import browserMetrics from '../browser-ml/oof-summary.json'

gsap.registerPlugin(ScrollTrigger)

const before = new URL('./assets/cabinet.svg', import.meta.url).href
const after = new URL('./assets/cabinet-after.svg', import.meta.url).href
const $ = (selector) => document.querySelector(selector)
const panel = $('#actions')
const comparison = $('#comparison')
const scene = $('#scene-img')
const viewport = $('#viewport')
const signals = {
  mp: { code: 'МП', title: 'Механическое повреждение', action: 'Проверить дверь, крепления и закрывание' },
  paint: { code: 'ЛКП', title: 'Повреждение покрытия', action: 'Восстановить защитное покрытие' },
}
const views = {
  corner: 'Нижний угол под углом 45°',
  hinges: 'Петли и крепления двери крупным планом',
  front: 'Дверь спереди',
  side: 'Боковая панель крупным планом',
}
const stages = ['inspect', 'rephoto', 'work', 'accept']
const guide = {
  inspect: ['mascot.svg', 'Сверьте дверь и покрытие с кадром. Каждый признак можно подтвердить отдельно.'],
  rephoto: ['mascot-retake.svg', 'Укажите, чего не видно. Я сохраню запрос нового ракурса в этой карточке.'],
  work: ['mascot.svg', 'Отметьте выполненные работы и приложите контрольный кадр.'],
  accept: ['mascot-accept.svg', 'Сравните до и после. Оставьте замечание открытым, если дефект виден.'],
}
const storageKey = 'mgts-showcase-scenario-v1'
const makeState = () => ({
  stage: 'inspect',
  inspection: { mp: null, paint: null },
  completed: { mp: false, paint: false },
  request: null,
  followup: null,
  after: null,
  acceptance: { mp: null, paint: null },
})
const memoryPhotos = { followup: null, after: null }
let selected = null
let showingAfter = false
let state = makeState()
let photoError = ''

try {
  const saved = JSON.parse(localStorage.getItem(storageKey))
  if (saved && stages.includes(saved.stage)) {
    state = { ...state, ...saved }
    if (!saved.completed && saved.after) {
      for (const key of Object.keys(signals)) state.completed[key] = state.inspection[key] === 'queued'
    }
    if (!saved.after) state.acceptance = { mp: null, paint: null }
  }
} catch {
  // The demonstration also works in browsers with storage disabled.
}

function save() {
  try { localStorage.setItem(storageKey, JSON.stringify(state)) }
  catch { /* User choices stay in memory if storage is blocked. */ }
}

function announce(message) {
  $('#announcement').textContent = message
}

function chapter(number, title, text) {
  return `<div class="eyebrow">ЭТАП ${number} / 04</div><h3 id="stage-title" tabindex="-1">${title}</h3><p>${text}</p>`
}

function signalCard(key, group) {
  const signal = signals[key]
  const isAcceptance = group === 'acceptance'
  const [yes, no] = isAcceptance ? ['confirmed', 'kept'] : ['queued', 'dismissed']
  return `<article class="signal-card" data-signal="${key}">
    <div class="signal-title"><span class="code ${key}">${signal.code}</span><strong>${signal.title}</strong></div>
    <p>${isAcceptance ? 'Оценка результата по фото после работ.' : signal.action}</p>
    <div class="decision" role="group" aria-label="${signal.code} — ${isAcceptance ? 'приёмка' : 'действие'}">
      <button type="button" class="button choice" data-decision="${group}" data-key="${key}" data-value="${yes}" aria-pressed="${state[group][key] === yes}">${isAcceptance ? 'Устранён' : 'Подтвердить'}</button>
      <button type="button" class="button choice" data-decision="${group}" data-key="${key}" data-value="${no}" aria-pressed="${state[group][key] === no}">${isAcceptance ? 'Оставить открытым' : 'Отклонить'}</button>
    </div>
  </article>`
}

function inspect() {
  const queued = Object.values(state.inspection).filter((value) => value === 'queued').length
  return chapter('01', 'Осмотр шкафа', 'Проверьте оба признака. У каждого своё действие и решение мастера.') +
    `<div class="action-grid">${Object.keys(signals).map((key) => signalCard(key, 'inspection')).join('')}</div>
    <div class="stage-foot">
      <span class="stage-status ${queued ? 'done' : ''}">Подтверждено действий — ${queued}</span>
      <button type="button" class="button primary" data-next="rephoto">К пересъёмке <span aria-hidden="true">→</span></button>
    </div>`
}

function uploadButtons(target) {
  return `<div class="upload">
    <button type="button" class="button primary" data-sample="${target}">${target === 'after' ? 'Приложить учебное фото' : 'Добавить учебный ракурс'}</button>
    <label class="button">
      Загрузить фото
      <input class="sr-only" type="file" accept="image/*" data-upload="${target}" aria-label="Загрузить собственное фото">
    </label>
  </div>
  <p class="error" id="upload-error" role="alert" ${photoError ? '' : 'hidden'}>${photoError}</p>`
}

function rephoto() {
  const title = chapter('02', 'Запросить нужный ракурс', 'Уточните, какую поверхность нужно показать крупнее. Запрос привязан к ТКШ-014.')
  if (!state.request) {
    return title + `<form id="request-form">
      <label class="field">Какой признак уточнить
        <select name="signal" required>
          <option value="mp">МП — механическое повреждение</option>
          <option value="paint">ЛКП — повреждение покрытия</option>
        </select>
      </label>
      <label class="field">Как переснять
        <select name="view" required>
          ${Object.entries(views).map(([key, label]) => `<option value="${key}">${label}</option>`).join('')}
        </select>
      </label>
      <div class="stage-foot"><button class="text-button" type="button" data-next="work">Ракурс достаточен</button><button class="button primary" type="submit">Создать запрос <span aria-hidden="true">→</span></button></div>
    </form>`
  }
  const { signal, view } = state.request
  const hasPhoto = Boolean(state.followup || memoryPhotos.followup)
  return title + `<div class="receipt"><strong>ТКШ-014 · ${signals[signal].code}</strong><br>${views[view]}<br>${hasPhoto ? 'Новый кадр прикреплён к карточке.' : 'Ожидает новый снимок.'}</div>
    ${uploadButtons('followup')}
    <div class="stage-foot">
      <button class="text-button" type="button" data-request-reset>Изменить запрос</button>
      <button class="button ${hasPhoto ? 'primary' : ''}" type="button" data-next="work">К работам <span aria-hidden="true">→</span></button>
    </div>`
}

function assignedKeys() {
  return Object.keys(signals).filter(key => state.inspection[key] === 'queued')
}

function works() {
  const keys = assignedKeys()
  const title = chapter('03', 'Работы мастера', 'В заказ попадают только действия, подтверждённые при осмотре.')
  if (!keys.length) return title + '<p class="receipt">Пока нет назначенных работ.</p><button type="button" class="button" data-next="inspect">Вернуться к осмотру</button>'
  const checks = keys.map(key => `<label class="work-check"><input type="checkbox" data-work="${key}" ${state.completed[key] ? 'checked' : ''}><span><strong>${signals[key].code}</strong> · ${signals[key].action}</span></label>`).join('')
  const hasPhoto = Boolean(state.after || memoryPhotos.after)
  return title + checks + (keys.some(key => state.completed[key])
    ? `<p class="receipt">${hasPhoto ? 'Контрольный кадр приложен.' : 'Приложите снимок после работ.'}</p>${uploadButtons('after')}
      <div class="stage-foot"><button class="button primary" type="button" data-next="accept" ${hasPhoto ? '' : 'disabled'}>К приёмке →</button></div>`
    : '<p class="receipt">Отметьте выполненную работу, чтобы приложить контрольный кадр.</p>')
}

function acceptance() {
  const keys = assignedKeys()
  const title = chapter('04', 'Приёмка ремонта', 'Сравните снимки и подтвердите результат по каждому выполненному действию.')
  if (!keys.length) return title + '<p class="receipt">Подтвердите хотя бы одно действие на этапе осмотра.</p><button type="button" class="button primary" data-next="inspect">К осмотру</button>'
  const ready = keys.filter(key => state.completed[key])
  if (!ready.length || !(state.after || memoryPhotos.after)) {
    return title + '<p class="receipt">Для приёмки нужны отметка мастера и снимок после работ.</p><button class="button primary" type="button" data-next="work">К работам</button>'
  }
  const open = keys.length - ready.filter(key => state.acceptance[key] === 'confirmed').length
  return title + `<div class="action-grid">${ready.map(key => signalCard(key, 'acceptance')).join('')}</div>
    <div class="stage-foot"><span class="stage-status ${open ? '' : 'done'}">${open ? 'Открытых признаков — ' + open : 'Все назначенные работы приняты'}</span></div>`
}

function displayPhoto() {
  const followup = state.stage === 'rephoto' && (state.followup || memoryPhotos.followup)
  const repair = ['work', 'accept'].includes(state.stage) && (state.after || memoryPhotos.after)
  const userPhoto = (repair && showingAfter && memoryPhotos.after) || (followup && memoryPhotos.followup)
  scene.src = userPhoto || (repair && showingAfter ? after : before)
  viewport.dataset.photo = userPhoto ? 'user' : followup ? 'close' : 'original'
  viewport.dataset.view = followup ? state.request.view : ''
  scene.alt = userPhoto
    ? 'Загруженный снимок шкафа'
    : repair && showingAfter ? 'Учебный снимок шкафа после работ' : 'Учебный снимок шкафа с двумя видимыми повреждениями'
  $('#image-tag').textContent = repair && showingAfter ? 'ПОСЛЕ РЕМОНТА' : followup ? 'УТОЧНЯЮЩИЙ КАДР' : 'ИСХОДНЫЙ СНИМОК'
  $('#photo-caption').textContent = userPhoto
    ? 'Пользовательское фото · доступно в этой вкладке'
    : followup ? 'Учебный фрагмент исходного изображения' : 'Синтетическое изображение для демонстрации'
  comparison.hidden = !repair
  for (const button of comparison.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.photo === (showingAfter ? 'after' : 'before')))
  }
  for (const button of viewport.querySelectorAll('[data-pin]')) {
    button.hidden = Boolean(userPhoto || (repair && showingAfter))
    button.setAttribute('aria-pressed', String(button.dataset.pin === selected))
  }
}

function render() {
  for (const button of document.querySelectorAll('[data-stage]')) {
    if (button.dataset.stage === state.stage) button.setAttribute('aria-current', 'step')
    else button.removeAttribute('aria-current')
  }
  const [asset, tip] = guide[state.stage]
  $('#guide-image').src = import.meta.env.BASE_URL + 'assets/' + asset
  $('#guide-tip').textContent = tip
  panel.innerHTML = { inspect, rephoto, work: works, accept: acceptance }[state.stage]()
  if (selected) panel.querySelector(`[data-signal="${selected}"]`)?.classList.add('is-selected')
  displayPhoto()
}

function stageTo(stage, focus = false) {
  state.stage = stage
  showingAfter = false
  selected = null
  photoError = ''
  save()
  render()
  if (focus) $('#stage-title').focus({ preventScroll: true })
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    gsap.fromTo(panel, { y: 5, opacity: .65 }, { y: 0, opacity: 1, duration: .27, ease: 'power2.out', overwrite: true })
  }
}

function rememberAttachment(target, url = null) {
  if (memoryPhotos[target]) URL.revokeObjectURL(memoryPhotos[target])
  memoryPhotos[target] = url
  state[target] = url ? null : 'sample'
  if (target === 'after') {
    showingAfter = true
    state.acceptance = { mp: null, paint: null }
  }
  photoError = ''
  save()
  render()
  $('#stage-title').focus({ preventScroll: true })
  announce(target === 'after' ? 'Фото после ремонта прикреплено к ТКШ-014' : 'Новый ракурс прикреплён к ТКШ-014')
}

for (const button of document.querySelectorAll('[data-stage]')) {
  button.addEventListener('click', () => stageTo(button.dataset.stage))
}

function reset() {
  for (const target of Object.keys(memoryPhotos)) {
    if (memoryPhotos[target]) URL.revokeObjectURL(memoryPhotos[target])
    memoryPhotos[target] = null
  }
  state = makeState()
  stageTo('inspect')
  announce('Учебный сценарий сброшен')
}

$('#reset').addEventListener('click', reset)
$('#reset-mobile').addEventListener('click', reset)

comparison.addEventListener('click', (event) => {
  const button = event.target.closest('[data-photo]')
  if (!button) return
  showingAfter = button.dataset.photo === 'after'
  displayPhoto()
  announce(showingAfter ? 'Показано фото после ремонта' : 'Показано исходное фото')
})

viewport.addEventListener('click', (event) => {
  const pin = event.target.closest('[data-pin]')
  if (!pin) return
  selected = selected === pin.dataset.pin ? null : pin.dataset.pin
  for (const card of panel.querySelectorAll('[data-signal]')) {
    card.classList.toggle('is-selected', card.dataset.signal === selected)
  }
  displayPhoto()
  if (selected) announce(signals[selected].title)
})

panel.addEventListener('click', (event) => {
  const button = event.target.closest('button')
  if (!button) return
  const { next, decision, key, value, sample, requestReset } = button.dataset
  if (next) stageTo(next, true)
  if (decision) {
    state[decision][key] = value
    if (decision === 'inspection' && value === 'dismissed') {
      state.acceptance[key] = null
      state.completed[key] = false
    }
    selected = key
    save()
    render()
    panel.querySelector(`[data-decision="${decision}"][data-key="${key}"][data-value="${value}"]`).focus({ preventScroll: true })
    announce(`${signals[key].code} · ${button.textContent}`)
  }
  if (sample) rememberAttachment(sample)
  if (requestReset !== undefined) {
    state.request = null
    state.followup = null
    if (memoryPhotos.followup) URL.revokeObjectURL(memoryPhotos.followup)
    memoryPhotos.followup = null
    photoError = ''
    save()
    render()
    panel.querySelector('select').focus({ preventScroll: true })
  }
})

panel.addEventListener('submit', (event) => {
  if (event.target.id !== 'request-form') return
  event.preventDefault()
  const form = new FormData(event.target)
  state.request = { signal: form.get('signal'), view: form.get('view') }
  selected = state.request.signal
  save()
  render()
  panel.querySelector('[data-sample]').focus({ preventScroll: true })
  announce('Запрос на пересъёмку создан для ТКШ-014')
})

panel.addEventListener('change', async (event) => {
  const input = event.target
  if (input.dataset.work) {
    const key = input.dataset.work
    state.completed[key] = input.checked
    state.acceptance[key] = null
    save()
    render()
    panel.querySelector(`[data-work="${key}"]`).focus({ preventScroll: true })
    return
  }
  if (!input.dataset.upload || !input.files?.[0]) return
  const file = input.files[0]
  const reportError = (message) => {
    photoError = message
    const error = $('#upload-error')
    if (!error) return
    error.hidden = false
    error.textContent = message
  }
  if (!file.type.startsWith('image/') || file.size > 6 * 1024 * 1024) {
    reportError('Выберите снимок до 6 МБ.')
    return
  }
  const url = URL.createObjectURL(file)
  const preview = new Image()
  preview.src = url
  try {
    await preview.decode()
    rememberAttachment(input.dataset.upload, url)
  } catch {
    URL.revokeObjectURL(url)
    reportError('Браузер не смог открыть это изображение.')
  }
})

render()

const formatPercent = new Intl.NumberFormat('ru-RU', { style: 'percent', maximumFractionDigits: 1 })
const precisePercent = new Intl.NumberFormat('ru-RU', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 })
const hybrid = vertex.runs['42'].blend_full_with_fallback
const browserRun = browserMetrics.seeds[browserMetrics.primary_seed].local_mean
for (const [mode, value] of [
  ['Сервер v0.4 + Gemini 4 Argon', hybrid.defect_macro_f1],
  ['Сервер v0.4 без облака', metrics.metrics.defect_macro_f1],
  ['Браузер · DINOv2 INT8', browserRun.defect_macro_f1],
]) {
  const row = document.createElement('tr')
  const name = document.createElement('th')
  name.scope = 'row'
  name.textContent = mode
  const score = document.createElement('td')
  score.textContent = precisePercent.format(value)
  row.append(name, score)
  $('#modes').append(row)
}
const cloudDamage = hybrid.any_damage
$('#hybrid-check').textContent = `В гибриде 75% оценки даёт локальная модель, 25% — Gemini. Найдено ${cloudDamage.tp} из ${cloudDamage.tp + cloudDamage.fn} кадров с повреждением. Ложных тревог — ${cloudDamage.fp} на ${cloudDamage.fp + cloudDamage.tn} исправных кадрах. При таймауте использована локальная оценка.`
const damage = metrics.metrics.any_damage
$('#validation-context').textContent = `${metrics.labeled} снимок с разметкой · ${metrics.groups} группы · ${metrics.folds} внешних фолдов`
for (const [label, value] of [
  ['Macro-F1 двух дефектов', formatPercent.format(metrics.metrics.defect_macro_f1)],
  ['Полнота по любому дефекту', formatPercent.format(damage.recall)],
  ['Найдено кадров с дефектом', `${damage.tp} из ${damage.tp + damage.fn}`],
  ['Верно определён тип шкафа', `${metrics.model.type_classifier.candidate_correct} из ${metrics.labeled}`],
]) {
  const cell = document.createElement('div')
  const term = document.createElement('dt')
  const result = document.createElement('dd')
  term.textContent = label
  result.textContent = value
  cell.append(term, result)
  $('#validation-results').append(cell)
}
$('#model-note').textContent = `Пропущено кадров с дефектом — ${damage.fn}. Ложных тревог — ${damage.fp}. Окончательное решение принимает мастер.`

$('#export-trip').addEventListener('click', () => {
  const inspection = { queued: 'Назначено', dismissed: 'Отклонено' }
  const acceptance = { confirmed: 'Устранён', kept: 'Оставлен открытым' }
  const rows = [['Поездка', 'Объект', 'Признак', 'Решение при осмотре', 'Работа', 'Выполнение', 'Запрос ракурса', 'Уточняющий кадр', 'Кадр после работ', 'Приёмка']]
  for (const [key, signal] of Object.entries(signals)) {
    const assigned = state.inspection[key] === 'queued'
    const requested = state.request?.signal === key
    rows.push(['Учебная · Москва', 'ТКШ-014', signal.title, inspection[state.inspection[key]] || 'Не рассмотрено',
      assigned ? signal.action : '—', assigned ? (state.completed[key] ? 'Выполнено' : 'Открыто') : '—',
      requested ? views[state.request.view] : '—', requested && (state.followup || memoryPhotos.followup) ? 'Есть' : '—',
      state.after || memoryPhotos.after ? 'Есть' : '—', acceptance[state.acceptance[key]] || '—'])
  }
  const csv = rows.map(row => row.map(value => '"' + value.replaceAll('"', '""') + '"').join(';')).join('\r\n')
  const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = 'Контур-учебная-поездка.csv'
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
  announce('Реестр учебной поездки подготовлен')
})

let opinionLoaded = false
$('#opinion').addEventListener('toggle', async event => {
  if (!event.currentTarget.open || opinionLoaded) return
  const output = $('#opinion-result')
  output.textContent = 'Читаем сохранённый ответ…'
  try {
    const response = await fetch(import.meta.env.BASE_URL + 'registry/argon-example.json')
    if (!response.ok) throw new Error('unavailable')
    const example = await response.json()
    const seconds = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 3 }).format(example.seconds)
    const type = example.scores.telecom >= 0.5 ? 'ТКШ' : 'РЩ'
    output.textContent = `Тип шкафа — ${type} (оценка ТКШ — ${formatPercent.format(example.scores.telecom)}). Механическое повреждение — ${formatPercent.format(example.scores.mechanical)}. Дефект покрытия — ${formatPercent.format(example.scores.coating)}. Время запроса — ${seconds} с.`
    opinionLoaded = true
  } catch {
    output.textContent = 'Не удалось прочитать запись. Закройте и откройте этот блок ещё раз.'
  }
})

$('[data-open-case]').addEventListener('click', () => stageTo('inspect'))
gsap.matchMedia().add('(min-width: 681px) and (prefers-reduced-motion: no-preference)', () => {
  gsap.from('.hero-copy', { y: 10, duration: .45, ease: 'power2.out', clearProps: 'transform' })
  gsap.to('.hero-scene', { yPercent: 4, ease: 'none', scrollTrigger: { trigger: '.night', start: 'top top', end: 'bottom top', scrub: true } })
})
