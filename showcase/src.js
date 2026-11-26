import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import metrics from './metrics.json'

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
const stages = ['inspect', 'rephoto', 'accept']
const storageKey = 'mgts-showcase-scenario-v1'
const makeState = () => ({
  stage: 'inspect',
  inspection: { mp: null, paint: null },
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
  if (saved && stages.includes(saved.stage)) state = { ...state, ...saved }
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
  return `<div class="eyebrow">ЭТАП ${number} / 03</div><h3 id="stage-title" tabindex="-1">${title}</h3><p>${text}</p>`
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
      <button type="button" class="button primary" data-next="rephoto">Дальше <span aria-hidden="true">→</span></button>
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
      <div class="stage-foot"><button class="button primary" type="submit">Создать запрос <span aria-hidden="true">→</span></button></div>
    </form>`
  }
  const { signal, view } = state.request
  const hasPhoto = Boolean(state.followup || memoryPhotos.followup)
  return title + `<div class="receipt"><strong>ТКШ-014 · ${signals[signal].code}</strong><br>${views[view]}<br>${hasPhoto ? 'Новый кадр прикреплён к карточке.' : 'Ожидает новый снимок.'}</div>
    ${uploadButtons('followup')}
    <div class="stage-foot">
      <button class="text-button" type="button" data-request-reset>Изменить запрос</button>
      <button class="button ${hasPhoto ? 'primary' : ''}" type="button" data-next="accept">К приёмке <span aria-hidden="true">→</span></button>
    </div>`
}

function acceptance() {
  const keys = Object.keys(signals).filter((key) => state.inspection[key] === 'queued')
  const title = chapter('03', 'Приёмка ремонта', 'Сравните снимки и подтвердите результат по каждому назначенному действию.')
  if (!keys.length) {
    return title + '<div class="receipt">Подтвердите хотя бы одно действие на этапе осмотра.</div><button type="button" class="button primary" data-next="inspect">К осмотру</button>'
  }
  const hasPhoto = Boolean(state.after || memoryPhotos.after)
  const confirmed = keys.filter((key) => state.acceptance[key] === 'confirmed').length
  const status = confirmed === keys.length
    ? 'Все назначенные работы приняты'
    : `Открытых признаков — ${keys.length - confirmed}`
  return title + (hasPhoto
    ? `<div class="action-grid">${keys.map((key) => signalCard(key, 'acceptance')).join('')}</div>
      <div class="stage-foot"><span class="stage-status ${confirmed === keys.length ? 'done' : ''}">${status}</span></div>`
    : `<div class="receipt">Карточка ожидает снимок после работ. После добавления фото станут доступны отдельные решения по двум признакам.</div>${uploadButtons('after')}`)
}

function displayPhoto() {
  const followup = state.stage === 'rephoto' && (state.followup || memoryPhotos.followup)
  const repair = state.stage === 'accept' && (state.after || memoryPhotos.after)
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
  panel.innerHTML = { inspect, rephoto, accept: acceptance }[state.stage]()
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
    if (decision === 'inspection' && value === 'dismissed') state.acceptance[key] = null
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
$('#validation-context').textContent = `${metrics.labeled} фото с разметкой · ${metrics.groups} групп похожих снимков · ${metrics.folds} фолдов`
for (const [label, value] of [
  ['Macro-F1 по МП и ЛКП', metrics.defectMacroF1],
  ['Macro-F1 базовой модели', metrics.baselineMacroF1],
  ['Полнота по любому дефекту', metrics.damageRecall],
  ['С дефектом среди приоритетных', `${metrics.priorityDamage} из ${metrics.priorityTotal}`],
]) {
  const item = document.createElement('div')
  const term = document.createElement('dt')
  const result = document.createElement('dd')
  term.textContent = label
  result.textContent = typeof value === 'number' ? formatPercent.format(value) : value
  item.append(term, result)
  $('#validation-results').append(item)
}

gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
  gsap.from('.intro h1, .intro-aside', { opacity: 0, y: 16, duration: .67, stagger: .075, clearProps: 'all', ease: 'power3.out' })
  gsap.from('.method-detail', {
    opacity: .75, y: 18, duration: .65, ease: 'power3.out',
    scrollTrigger: { trigger: '.method', start: 'top 86%', once: true },
  })
})
