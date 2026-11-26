import {condition,recommendation} from './order-state.js'
import {assessInputQuality} from './input-quality.js'
import {allAfter, croppedCanvas, downloadReport, evidenceId, previousRetakes, receiptFor, regionName, strongestRegion} from './evidence.js'

const DATABASE = 'mgts-browser-work-orders'
const STORE = 'orders'
const SIGNALS = {
  mechanical: { code: 'МП', title: 'Механическое повреждение', task: 'Проверить дверь, крепления и закрывание' },
  coating: { code: 'ЛКП', title: 'Повреждение покрытия', task: 'Восстановить защитное покрытие' },
}
const VIEWS = {
  hinges: 'Петли и крепления двери крупным планом',
  front: 'Дверь спереди',
  side: 'Боковая панель крупным планом',
  corner: 'Нижний угол под углом 45°',
  coating: 'Локальный участок покрытия при рассеянном свете',
}
const EMPTY = () => ({ decision: 'pending', result: null })
const sessions = new WeakMap()
const memory = new Map()
const percent = new Intl.NumberFormat('ru-RU', { style: 'percent', maximumFractionDigits: 1 })
const date = new Intl.DateTimeFormat('ru-RU', { dateStyle: 'short', timeStyle: 'short' })
let databasePromise

function db() {
  if (!databasePromise) databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'id' })
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
    request.onblocked = () => reject(new Error('Локальное хранилище занято'))
  })
  return databasePromise
}

async function transaction(mode, operation) {
  const database = await db()
  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE, mode)
    const request = operation(tx.objectStore(STORE))
    tx.oncomplete = () => resolve(request.result)
    tx.onerror = () => reject(tx.error)
    tx.onabort = () => reject(tx.error)
  })
}

async function loadHistory() {
  try {
    const records = await transaction('readonly', (store) => store.getAll())
    for (const entry of records) {
      const current = memory.get(entry.id)
      if (!current || entry.updatedAt > current.updatedAt) memory.set(entry.id, entry)
    }
  } catch {
    // The session can still be used when browser storage is unavailable.
  }
  return [...memory.values()].sort((a, b) => b.updatedAt - a.updatedAt)
}

async function persist(host) {
  host.order.updatedAt = Date.now()
  const record = structuredClone(host.order)
  memory.set(record.id, record)
  try {
    await transaction('readwrite', (store) => store.put(record))
    host.persisted = true
  } catch {
    host.persisted = false
  }
}

function node(tag, className = '', text = '') {
  const element = document.createElement(tag)
  if (className) element.className = className
  if (text) element.textContent = text
  return element
}

function button(text, listener, variant = '') {
  const element = node('button', `wo-button ${variant}`, text)
  element.type = 'button'
  element.addEventListener('click', listener)
  return element
}

function pressedButton(text, enabled, listener, variant = '') {
  const element = button(text, listener, variant)
  element.setAttribute('aria-pressed', String(enabled))
  return element
}

function field(title, control) {
  const label = node('label', 'wo-field')
  label.append(node('span', '', title), control)
  return label
}

function choose(options) {
  const select = node('select')
  for (const [key, value] of Object.entries(options)) {
    const option = node('option', '', value)
    option.value = key
    select.append(option)
  }
  return select
}

function selectedEvidence(host) {
  const { order, photo } = host
  if (host.archiveId) {
    const photos=[...allAfter(order),...previousRetakes(order).map(r=>r.evidence).filter(Boolean)]
    const selected=photos.find(p=>evidenceId(p)===host.archiveId)
    if (selected) return selected
  }
  return photo === 'rephoto' && order.rephoto ? order.rephoto
    : photo === 'after' && order.after ? order.after
      : order.before
}

async function mutate(host, update) {
  if (sessions.get(host.container) !== host) return
  update(host.order)
  persist(host).then(() => {
    const note = host.container.querySelector('[data-storage-note]')
    if (note) note.textContent = storageNote(host)
  })
  await render(host)
}

function storageNote(host) {
  return host.persisted
    ? 'Карточка и фото сохранены в этом браузере.'
    : 'Фото доступны в этой вкладке. Браузер мог запретить локальное сохранение.'
}

function photoPane(host) {
  const pane = node('div', 'wo-photo')
  const evidence = selectedEvidence(host)
  if (evidence.image instanceof Blob) {
    const image = node('img')
    image.src = URL.createObjectURL(evidence.image)
    image.alt = `Карточка ${host.order.id} · ${evidence.caption}`
    host.urls.push(image.src)
    const bounds=host.region && evidence===host.order.before && strongestRegion(host.order,host.region)
    if (bounds) {
      const [x0,y0,x1,y1]=bounds
      const stage=node('div','wo-region-stage'),frame=node('span','wo-region-box')
      frame.style.cssText=`left:${x0*100}%;top:${y0*100}%;width:${(x1-x0)*100}%;height:${(y1-y0)*100}%`
      stage.append(image,frame)
      const crop=node('div','wo-crop')
      image.addEventListener('load',()=>{
        if(!stage.isConnected)return
        const canvas=croppedCanvas(image,bounds)
        canvas.setAttribute('aria-label','Увеличенный фрагмент')
        crop.replaceChildren(canvas)
      },{once:true})
      pane.append(stage,node('p','wo-muted',SIGNALS[host.region].code+' · '+regionName(bounds)+'. Из четырёх фрагментов этот сильнее всего повлиял на оценку признака.'),crop)
    } else pane.append(image)
  }
  if (evidence!==host.order.before && evidence.image && host.order.before?.image) {
    const compare=node('details','wo-compare')
    compare.append(node('summary','','Сравнить с исходным кадром'))
    const pair=node('div','wo-pair')
    for (const [e,label] of [[host.order.before,'До'],[evidence,evidence.caption]]) {
      const figure=node('figure'),img=node('img')
      img.src=URL.createObjectURL(e.image);host.urls.push(img.src);img.alt=label
      figure.append(img,node('figcaption','',label));pair.append(figure)
    }
    compare.append(pair);pane.append(compare)
  }
  const footer = node('div', 'wo-photo-foot')
  footer.append(node('strong', '', evidence.caption), node('span', 'wo-filename', evidence.filename))
  const tabs = node('div', 'wo-buttons wo-photo-tabs')
  tabs.setAttribute('role', 'group')
  tabs.setAttribute('aria-label', 'Снимки объекта')
  for (const [key, caption] of [['before', 'До'], ['rephoto', 'Пересъёмка'], ['after', 'После']]) {
    if (!host.order[key]) continue
    tabs.append(pressedButton(caption, host.photo === key, () => {
      host.photo = key; host.archiveId=null;host.region=null
      render(host)
    }))
  }
  pane.append(footer, tabs)
  return pane
}

function signalPane(host) {
  const pane = node('div', 'wo-signals')
  pane.append(node('h3', '', 'Работы по каждому признаку'))
  for (const [key, signal] of Object.entries(SIGNALS)) {
    const state = host.order.signals[key]
    const score = host.order.scores[key]
    const card = node('article', 'wo-signal')
    const heading = node('div', 'wo-signal-title')
    heading.append(node('span', `wo-signal-code wo-${key}`, signal.code), node('h4', '', signal.title))
    if (Number.isFinite(score)) heading.append(node('span', 'wo-score', percent.format(score)))
    card.append(heading, node('p', '', signal.task))
    if (strongestRegion(host.order,key)) card.append(pressedButton('Показать фрагмент',host.region===key,()=>{
      host.region=host.region===key?null:key;host.archiveId=null;host.photo='before';render(host)
    }))
    const actions = node('div', 'wo-buttons')
    actions.setAttribute('aria-label', signal.title)
    const set = (decision) => mutate(host, (order) => {
      if (order.signals[key].decision !== decision) order.signals[key] = { decision, result: null }
    })
    const label = Number.isFinite(score) && score < .5 ? 'Добавить в работы' : 'Подтвердить'
    actions.append(
      pressedButton(label, state.decision === 'confirmed', () => set('confirmed')),
      pressedButton('Снять признак', state.decision === 'dismissed', () => set('dismissed')),
    )
    for (const [index, control] of [...actions.children].entries()) control.dataset.focusKey = `${key}-decision-${index}`
    card.append(actions)
    const usable=allAfter(host.order).filter(e=>e.orderId===host.order.id && e.identityConfirmed && (!e.scope||e.scope==='object'||e.scope===key))
    if (usable.length && state.decision === 'confirmed') {
      const acceptance=node('div','wo-acceptance')
      const receipt=receiptFor(host.order,key)
      const chosen=receipt || usable.at(-1)
      const select=choose(Object.fromEntries(usable.map((e,i)=>[evidenceId(e),`${i+1}. ${e.filename}`])))
      select.value=evidenceId(chosen)
      acceptance.append(field('Снимок для приёмки '+signal.code,select))
      const options=node('div','wo-buttons')
      const inspect=result=>{
        const active=usable.find(e=>evidenceId(e)===select.value)
        if (!active)return
        return mutate(host,order=>{
          Object.assign(order.signals[key],{result,evidenceId:evidenceId(active),reviewedAt:Date.now()})
          host.photo='after';host.archiveId=evidenceId(active);host.region=null
        })
      }
      options.append(pressedButton('Устранён',state.result==='resolved',()=>inspect('resolved'),'wo-success'),
        pressedButton('Оставить открытым',state.result==='retained',()=>inspect('retained'),'wo-warning'))
      acceptance.append(options)
      if(receipt)acceptance.append(button('Открыть снимок решения · '+receipt.filename,()=>{
        host.photo='after';host.archiveId=evidenceId(receipt);host.region=null;render(host)
      }))
      card.append(acceptance)
    }
    pane.append(card)
  }
  const assigned = Object.values(host.order.signals).filter((signal) => signal.decision === 'confirmed')
  const accepted = assigned.filter((signal) => signal.result === 'resolved').length
  const status = assigned.length === 0 ? 'Выберите действия после просмотра исходного снимка.'
    : accepted === assigned.length ? 'Все назначенные работы приняты.'
      : `Назначено — ${assigned.length}. Открыто — ${assigned.length - accepted}.`
  pane.append(node('p', 'wo-summary', status))
  return pane
}

function requestPane(host) {
  const pane = node('div', 'wo-task')
  pane.append(node('h3', '', 'Прицельная пересъёмка'))
  const request = host.order.request
  const fulfilled = Boolean(host.order.rephoto)
  const history=host.order.retakeHistory || []
  if (history.length) {
    const prior=node('details','wo-compare')
    prior.append(node('summary','','Предыдущие пересъёмки · '+history.length))
    for(const {request:r,evidence:e} of history) prior.append(button(SIGNALS[r.signal].code+' · '+VIEWS[r.view]+' · '+e.filename,()=>{
      host.region=null;host.archiveId=evidenceId(e);host.photo='rephoto';render(host)
    }))
    pane.append(prior)
  }
  if (request && (!host.requestEditing || fulfilled)) {
    pane.append(node('p', 'wo-note', `${SIGNALS[request.signal].code} · ${VIEWS[request.view]}`))
    if (request.source === 'model') {
      const score = percent.format(request.score)
      pane.append(node('p', 'wo-muted', `Ракурс предложен по пограничной оценке модели (${score}).`))
    }
    if (strongestRegion(host.order,request.signal)) pane.append(button('Открыть область на исходном фото',()=>{
      host.photo='before';host.archiveId=null;host.region=request.signal;render(host)
    }))
    if (fulfilled) {
      pane.append(node('p', 'wo-muted', 'Запрос выполнен. Новый кадр сохранён в этой карточке.'),
        button('Запросить ещё один ракурс',()=>mutate(host,order=>{
          (order.retakeHistory ||= []).push({request:order.request,evidence:order.rephoto})
          order.request=null;order.rephoto=null;host.archiveId=null;host.photo='before'
        })))
    } else {
      pane.append(attachment(host, 'rephoto'))
      pane.append(button('Изменить запрос', () => {
        if (host.order.rephoto) return
        host.requestEditing = true
        render(host)
      }))
    }
    return pane
  }
  const form = node('form', 'wo-form')
  const signal = choose(Object.fromEntries(Object.entries(SIGNALS).map(([key, value]) => [key, `${value.code} — ${value.title.toLowerCase()}`])))
  const view = choose(VIEWS)
  if (request) {
    signal.value = request.signal
    view.value = request.view
  }
  const submit = node('button', 'wo-button wo-primary', request ? 'Сохранить запрос' : 'Создать запрос')
  submit.type = 'submit'
  form.append(field('Признак', signal), field('Нужный ракурс', view), submit)
  form.addEventListener('submit', (event) => {
    event.preventDefault()
    if (host.order.rephoto) return
    host.requestEditing = false
    mutate(host, (order) => { order.request = { signal: signal.value, view: view.value, source: 'manual' } })
  })
  if (request) form.append(button('Отменить изменение', () => {
    host.requestEditing = false
    render(host)
  }))
  pane.append(form)
  return pane
}

function acceptancePane(host) {
  const pane = node('div', 'wo-task')
  pane.append(node('h3', '', 'Фото после ремонта'))
  if (Object.values(host.order.signals).some((signal) => signal.decision === 'confirmed')) {
    pane.append(node('p', 'wo-muted', 'После сверки объекта можно отдельно принять каждую назначенную работу.'))
    if (host.order.after) pane.append(node('p', 'wo-note', 'Снимок прикреплён к этой карточке.'))
    pane.append(attachment(host, 'after'))
  } else {
    pane.append(node('p', 'wo-muted', 'Сначала подтвердите хотя бы одну работу.'))
  }
  return pane
}

async function sameFile(left, right) {
  if (left.size !== right.size) return false
  const [a, b] = await Promise.all([left.arrayBuffer(), right.arrayBuffer()])
  const leftBytes = new Uint8Array(a)
  const rightBytes = new Uint8Array(b)
  return leftBytes.every((byte, index) => byte === rightBytes[index])
}

async function attachmentIssue(image) {
  const url = URL.createObjectURL(image)
  const preview = new Image()
  try {
    preview.src = url
    await preview.decode()
    if (preview.naturalWidth * preview.naturalHeight > 40_000_000) {
      return 'Выберите фотографию до 40 мегапикселей.'
    }
    const quality = assessInputQuality(preview)
    return quality.ok ? null : quality.request
  } finally {
    URL.revokeObjectURL(url)
  }
}

async function belongsToAnotherOrder(image, orderId) {
  const records = await loadHistory()
  for (const record of records) {
    if (record.id === orderId) continue
    for (const evidence of [record.before,...allAfter(record),...previousRetakes(record).map(r=>r.evidence)]) {
      if (evidence?.image instanceof Blob && await sameFile(image, evidence.image)) return true
    }
  }
  return false
}

function attachment(host, kind) {
  const orderId = host.order.id
  const activeRequest = kind === 'rephoto' ? host.order.request : null
  const form = node('form', 'wo-form wo-attachment')
  const file = node('input')
  file.type = 'file'
  file.accept = 'image/jpeg,image/png,image/webp,image/avif,image/gif'
  file.required = true
  const confirm = node('input')
  confirm.type = 'checkbox'
  confirm.required = true
  const label = node('label', 'wo-identity')
  label.append(confirm, node('span', '', `Визуально сверил объект с исходным фото ${orderId}`))
  let scope
  if (kind==='after') {
    const choices={object:'Общий кадр работ'}
    for(const [key,sig] of Object.entries(SIGNALS))if(host.order.signals[key].decision==='confirmed')choices[key]=sig.title
    scope=choose(choices)
    form.append(field('Какая работа показана',scope))
  }
  const submitLabel = kind === 'after' ? 'Приложить фото после' : 'Приложить пересъёмку'
  const submit = node('button', 'wo-button wo-primary', submitLabel)
  submit.type = 'submit'
  const message = node('p', 'wo-error')
  message.setAttribute('role', 'alert')
  const preview = node('img', 'wo-upload-preview')
  preview.alt = 'Выбранный новый снимок для сверки объекта'
  preview.hidden = true
  form.append(field('Выберите снимок', file), preview, label, submit, message)
  file.addEventListener('change', () => {
    confirm.checked = false
    message.textContent = ''
    if (preview.src) URL.revokeObjectURL(preview.src)
    preview.removeAttribute('src')
    preview.hidden = true
    const image = file.files?.[0]
    if (!image?.type.startsWith('image/') || image.size > 12 * 1024 * 1024) return
    preview.src = URL.createObjectURL(image)
    host.urls.push(preview.src)
    preview.hidden = false
  })
  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    const image = file.files?.[0]
    if (host.order.id !== orderId || !confirm.checked || !image) return
    if (!/^image\/(jpeg|png|webp|avif|gif)$/.test(image.type) || image.size > 12 * 1024 * 1024) {
      message.textContent = 'Выберите JPEG, PNG, WebP, AVIF или GIF до 12 МБ.'
      return
    }
    submit.disabled = true
    message.textContent = ''
    try {
      const issue = await attachmentIssue(image)
      if (issue) {
        message.textContent = 'Нужна пересъёмка. ' + issue
        return
      }
      const previous=[host.order.before,...previousRetakes(host.order).map(r=>r.evidence),...allAfter(host.order)]
      for (const evidence of previous) {
        if (evidence?.image && await sameFile(image,evidence.image)) {
          message.textContent='Этот файл уже есть в карточке. Выберите новый снимок.'
          return
        }
      }
      if (await belongsToAnotherOrder(image, orderId)) {
        message.textContent = 'Этот снимок уже привязан к другой карточке. Приложите кадр текущего объекта.'
        return
      }
      if (host.order.id !== orderId || !form.isConnected) return
      if (kind === 'rephoto' && (host.order.request !== activeRequest || host.order.rephoto)) return
      await mutate(host, (order) => {
        if(kind==='after' && order.after) {
          for (const st of Object.values(order.signals))if(st.result && !st.evidenceId)st.evidenceId=evidenceId(order.after)
          order.afterHistory ||= []
          order.afterHistory.push(order.after)
        }
        order[kind] = {
          id:crypto.randomUUID(),createdAt:Date.now(),scope:scope?.value || activeRequest?.signal || 'object',
          orderId,
          identityConfirmed: true,
          filename: image.name,
          image,
          caption: kind === 'after' ? 'Фото после ремонта' : 'Запрошенный ракурс',
        }
        host.photo = kind;host.archiveId=null;host.region=null
      })
    } catch {
      message.textContent = 'Браузер не смог открыть это изображение.'
    } finally {
      submit.disabled = false
    }
  })
  return form
}

function coordinate(raw, maximum) {
  const text = raw.trim().replace(',', '.').replace('−', '-')
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) return null
  const value = Number(text)
  return Number.isFinite(value) && Math.abs(value) <= maximum ? value : null
}

function locationPane(host) {
  const pane = node('div', 'wo-task wo-location')
  const location = host.order.location
  pane.append(
    node('h3', '', 'Координаты шкафа'),
    node('p', 'wo-muted', location ? 'Источник координат · ручная привязка, WGS84' : 'Местоположение неизвестно. Введите широту и долготу шкафа в WGS84.'),
  )
  const identity = node('form','wo-form')
  const asset = node('input')
  asset.maxLength = 80
  asset.value = host.order.assetId || ''
  const save = node('button','wo-button','Сохранить ID')
  save.type='submit'
  const saved=node('span','wo-muted',host.order.assetId ? 'ID сохранён' : '')
  identity.append(field('ID оборудования из реестра',asset),save,saved)
  identity.onsubmit=async event=>{event.preventDefault();await mutate(host,order=>{order.assetId=asset.value.trim()})}
  pane.append(identity)
  const form = node('form', 'wo-form wo-coordinate-form')
  const latitude = node('input')
  const longitude = node('input')
  for (const input of [latitude, longitude]) {
    input.type = 'text'
    input.inputMode = 'decimal'
    input.autocomplete = 'off'
    input.maxLength = 24
  }
  latitude.name = 'latitude'
  latitude.placeholder = '55.751244'
  longitude.name = 'longitude'
  longitude.placeholder = '37.618423'
  latitude.value = location ? String(location.lat) : ''
  longitude.value = location ? String(location.lon) : ''
  const submit = node('button', 'wo-button wo-primary', 'Сохранить координаты')
  submit.type = 'submit'
  const message = node('p', 'wo-error')
  message.setAttribute('role', 'alert')
  form.append(field('Широта', latitude), field('Долгота', longitude), submit, message)
  form.addEventListener('submit', (event) => {
    event.preventDefault()
    const latRaw = latitude.value.trim()
    const lonRaw = longitude.value.trim()
    latitude.removeAttribute('aria-invalid')
    longitude.removeAttribute('aria-invalid')
    if (!latRaw && !lonRaw) {
      mutate(host, (order) => { order.location = null })
      return
    }
    const lat = coordinate(latRaw, 90)
    const lon = coordinate(lonRaw, 180)
    if (lat === null || lon === null) {
      if (lat === null) latitude.setAttribute('aria-invalid', 'true')
      if (lon === null) longitude.setAttribute('aria-invalid', 'true')
      message.textContent = 'Введите обе координаты. Широта от −90 до 90, долгота от −180 до 180.'
      return
    }
    mutate(host, (order) => { order.location = { lat, lon, source: 'manual' } })
  })
  pane.append(form)
  if (location) {
    const link = node('a', 'wo-button wo-map-link', 'Показать на карте ↗')
    const { lat, lon } = location
    link.href = `https://www.openstreetmap.org/?mlat=${encodeURIComponent(lat)}&mlon=${encodeURIComponent(lon)}#map=17/${encodeURIComponent(lat)}/${encodeURIComponent(lon)}`
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    pane.append(link)
  }
  return pane
}

function historyPane(host, records) {
  const history = node('details', 'wo-history')
  history.open = host.historyOpen
  history.addEventListener('toggle', () => { host.historyOpen = history.open })
  history.append(node('summary', '', `История осмотров · ${records.length}`))
  const list = node('ul')
  for (const record of records) {
    const item = node('li')
    const current = record.id === host.order.id
    const link = button(`${record.id} · ${record.object_type} · ${date.format(record.createdAt)}`, () => {
      host.order = structuredClone(record)
      host.archiveId=null;host.region=null
      host.photo = 'before'
      host.requestEditing = false
      render(host)
    })
    if (current) link.setAttribute('aria-current', 'true')
    item.append(link)
    list.append(item)
  }
  history.append(list)
  return history
}

async function render(host) {
  const renderId = (host.renderId ?? 0) + 1
  host.renderId = renderId
  const history = await loadHistory()
  if (host.renderId !== renderId || sessions.get(host.container) !== host) return
  const previousControl = document.activeElement
  const focusInside = host.container.contains(previousControl)
  for (const url of host.urls) URL.revokeObjectURL(url)
  host.urls = []
  const card = node('div', 'wo')
  const header = node('header', 'wo-header')
  const title = node('div')
  const titleHeading = node('h2', '', `${host.order.object_type} · ${host.order.id}`)
  titleHeading.tabIndex = -1
  title.append(node('span', 'wo-eyebrow', 'КАРТОЧКА ОСМОТРА'), titleHeading)
  title.append(node('p','wo-summary','Состояние по фото · '+condition(host.order)),node('p','wo-muted','Рекомендация · '+recommendation(host.order)))
  if(host.order.routeName)title.append(node('p','wo-muted','Рейс · '+host.order.routeName))
  if(host.order.assetId)title.append(node('p','wo-muted','Оборудование · '+host.order.assetId))
  const exportButton=button('Скачать акт осмотра',async()=>{
    exportButton.disabled=true
    try {await downloadReport(host.order,SIGNALS,VIEWS)}
    catch {const error=node('p','wo-error','Не удалось собрать акт. Проверьте доступность приложенных фото.');error.setAttribute('role','alert');header.append(error)}
    finally{exportButton.disabled=false}
  })
  header.append(title, historyPane(host, history),exportButton)
  const main = node('div', 'wo-main')
  main.append(photoPane(host), signalPane(host))
  const followUp = node('div', 'wo-followup')
  followUp.append(requestPane(host), acceptancePane(host))
  const note = node('p', 'wo-muted wo-storage', storageNote(host))
  note.dataset.storageNote = ''
  note.setAttribute('aria-live', 'polite')
  card.append(header, main, followUp, locationPane(host), note)
  host.container.hidden = false
  host.container.replaceChildren(card)
  if (focusInside) {
    const key = previousControl?.dataset?.focusKey
    const matching = key ? card.querySelector(`[data-focus-key="${CSS.escape(key)}"]`) : null
    const nextControl = matching ?? titleHeading
    nextControl.focus({ preventScroll: true })
  }
}

function mount(container, record) {
  const previous = sessions.get(container)
  if (previous) {
    previous.renderId = -1
    for (const url of previous.urls) URL.revokeObjectURL(url)
  }
  const host = { container, order: record, photo: 'before', urls: [], persisted: true, historyOpen: false, requestEditing: false }
  sessions.set(container, host)
  return host
}

function automaticRequest(scores) {
  let signal = null
  let nearest = Infinity
  for (const key of Object.keys(SIGNALS)) {
    const score = scores[key]
    if (!Number.isFinite(score) || score < 0.4 || score > 0.6) continue
    const distance = Math.abs(score - 0.5)
    if (distance >= nearest - Number.EPSILON) continue
    signal = key
    nearest = distance
  }
  if (!signal) return null
  const view = signal === 'mechanical' ? 'hinges' : 'coating'
  return { signal, view, source: 'model', score: scores[signal] }
}

export async function openOrder(container, { object_type, scores, filename, image, regions, model_version, routeName }) {
  const timestamp = Date.now()
  const id = `ОСМ-${crypto.randomUUID().slice(0, 8).toUpperCase()}`
  const safeScore = (value) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null
  const normalizedScores = { mechanical: safeScore(scores?.mechanical), coating: safeScore(scores?.coating) }
  const order = {
    id,
    object_type: String(object_type || 'Шкаф'),
    scores: normalizedScores, regions:regions||{}, model_version, routeName:routeName||'',
    retakeHistory:[],afterHistory:[],
    createdAt: timestamp,
    updatedAt: timestamp,
    before: { id:crypto.randomUUID(),createdAt:timestamp,filename: String(filename || 'Снимок'), image, caption: 'Исходный снимок' },
    signals: { mechanical: EMPTY(), coating: EMPTY() },
    request: automaticRequest(normalizedScores),
    rephoto: null,
    after: null,
    location: null,
  }
  const host = mount(container, order)
  await persist(host)
  await render(host)
  return order.id
}

export async function listOrders() {
  return loadHistory()
}

export async function showOrder(container,id) {
  const order=(await loadHistory()).find(entry=>entry.id===id)
  if (!order) return false
  await render(mount(container,structuredClone(order)))
  return true
}

export async function restoreOrder(container) {
  const [record] = await loadHistory()
  if (!record) return false
  await render(mount(container, structuredClone(record)))
  return true
}
