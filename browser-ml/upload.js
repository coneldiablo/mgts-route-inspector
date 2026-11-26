import {assessInputQuality} from './input-quality.js';
import {analyzeOnServer, cloudStatusText, loadEndpoint, modeNames} from './hybrid.js?v=20261126-hybrid1';
import {openOrder, restoreOrder} from './workorder.js?v=20261126-hybrid1';
import {displayRoutes, existingEntry, saveEntry} from './route-batch.js?v=20261126-hybrid1';

const form = document.querySelector('#input');
const status = document.querySelector('#status');
const box = document.querySelector('#order');
const disclosure = document.querySelector('#analysis-disclosure');
const options = document.querySelector('.analysis-options');
const serverDisclosure = disclosure.textContent;
const say = text => { status.textContent = text; };
options.addEventListener('change', () => {
  disclosure.textContent = form.elements['analysis-mode'].value === 'server'
    ? serverDisclosure
    : 'Фото остаются на этом устройстве. Браузер рассчитает тип шкафа и оба признака с помощью DINOv2.';
});

restoreOrder(box).catch(() => say('Хранилище браузера сейчас недоступно.'));
displayRoutes();

async function analyze(bitmap, batch, progress) {
  let fallbackReason = '';
  if (batch.mode === 'server') {
    if (batch.pauseReason) {
      fallbackReason = batch.pauseReason + ' Этот кадр не отправлен.';
    } else {
      try {
        batch.endpoint ??= await loadEndpoint();
        const result = await analyzeOnServer(bitmap, batch.endpoint, progress);
        if (result.mode === 'server_local') {
          result.modeDetail = cloudStatusText(result.analysis.cloud.status);
          if (['quota_exhausted','access_denied','adc_unavailable','model_unavailable'].includes(result.analysis.cloud.status)) {
            batch.pauseReason = result.modeDetail;
          }
        }
        return result;
      } catch (error) {
        if (error.rejectImage) throw error;
        fallbackReason = error.message;
        if (error.pauseBatch) batch.pauseReason = 'Полный анализ приостановлен для этого рейса.';
      }
    }
  }
  const detail = fallbackReason || 'Фото не отправлялось на сервер.';
  try {
    const {recognize} = await import('./infer.js');
    const result = await recognize(bitmap, text => progress(detail + ' ' + text));
    return {...result, mode: fallbackReason ? 'browser_fallback' : 'browser', modeDetail: detail};
  } catch {
    throw Error(detail + ' Браузерная модель недоступна. Проверьте соединение для загрузки модели.');
  }
}

async function processFile(file, route, batch, progress) {
  if (file.size > 20*1024*1024) throw Error('Выберите файл размером до 20 МБ. Фото не отправлено.');
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) throw Error('Поддерживаются JPEG, PNG и WebP. Файл не отправлен.');
  let hash = '';
  if (route) {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    hash = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2,'0')).join('');
    if (existingEntry(route, hash)) return {duplicate: true};
  }
  let bitmap;
  try { bitmap = await createImageBitmap(file); }
  catch { throw Error('Не удалось прочитать изображение. Фото не отправлено.'); }
  try {
    if (bitmap.width * bitmap.height > 40_000_000) throw Error('Выберите фотографию до 40 мегапикселей. Фото не отправлено.');
    const quality = assessInputQuality(bitmap);
    if (!quality.ok) return {file: file.name, hash, reason: 'Нужна пересъёмка. ' + quality.request + ' Фото не отправлено.'};
    const result = await analyze(bitmap, batch, progress);
    const objectType = result.object_type === 'telecom_cabinet' ? 'ТКШ' : 'РЩ';
    const orderId = await openOrder(box, {...result, object_type: objectType, filename: file.name, image: file, routeName: route});
    return {file: file.name, hash, orderId, objectType, mode: result.mode, modeDetail: result.modeDetail};
  } finally {
    bitmap.close();
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = document.querySelector('#analyze');
  const source = document.querySelector('#photo');
  const nameInput = document.querySelector('#route-name');
  const files = Array.from(source.files || []);
  if (!files.length || button.disabled) return;
  const batch = {mode: form.elements['analysis-mode'].value, endpoint: null, pauseReason: ''};
  let route = nameInput.value.trim();
  if (files.length > 1 && !route) {
    route = 'Выезд ' + new Date().toLocaleString('ru-RU') + ' · ' + crypto.randomUUID().slice(0,4);
    nameInput.value = route;
  }
  button.disabled = source.disabled = nameInput.disabled = options.disabled = true;
  let ready = 0, failed = 0, duplicates = 0, volatile = false;
  const modes = new Map();
  try {
    for (const [index, file] of files.entries()) {
      const progress = text => say((files.length > 1 ? `Кадр ${index+1} из ${files.length}. ` : '') + text);
      progress('Читаем фотографию…');
      const started = performance.now();
      let row;
      try { row = await processFile(file, route, batch, progress); }
      catch (error) { row = {file: file.name, reason: error.message || 'Не удалось завершить анализ.'}; }
      if (row.duplicate) { duplicates++; continue; }
      row.processingMs = Math.round(performance.now() - started);
      if (route && !saveEntry(route, row)) volatile = true;
      if (row.orderId) { ready++; modes.set(row.mode, (modes.get(row.mode) || 0) + 1); }
      else failed++;
      if (files.length === 1) say(row.orderId ? ['Осмотр готов.', modeNames[row.mode] + '.', row.modeDetail].filter(Boolean).join(' ') : row.reason);
    }
    if (route) displayRoutes(route, volatile);
    if (files.length > 1 || duplicates) {
      const counts = [...modes].map(([mode, count]) => modeNames[mode] + ' — ' + count).join('. ');
      say(`Обработка завершена. Новых карточек: ${ready}. На проверку файла или пересъёмку: ${failed}. Уже учтено в рейсе: ${duplicates}.` + (counts ? ' ' + counts + '.' : ''));
    }
    if (files.length === 1 && ready) box.scrollIntoView({block: 'start'});
  } finally {
    button.disabled = source.disabled = nameInput.disabled = options.disabled = false;
  }
});
