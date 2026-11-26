const maxJpegBytes = 5 * 1024 * 1024;
const knownObjectTypes = ['telecom_cabinet', 'distribution_board'];
const probability = value => Number.isFinite(value) && value >= 0 && value <= 1;
const validScores = scores => probability(scores?.mechanical) && probability(scores?.coating);
const named = value => typeof value === 'string' && value.trim().length > 0;

export const modeNames = {
  hybrid: 'Полный анализ · сервер и Google Vertex AI',
  server_local: 'Серверный анализ · только локальная модель',
  browser: 'Браузерный анализ · на устройстве',
  browser_fallback: 'Резервный анализ · на устройстве',
};

const cloudStatuses = {
  unreadable: 'Облачная модель запросила более читаемый снимок.',
  quota_exhausted: 'Исчерпана квота облачной модели.',
  adc_unavailable: 'Серверу недоступна авторизация Google Vertex AI.',
  access_denied: 'Google Vertex AI отклонил запрос.',
  model_unavailable: 'Облачная модель сейчас недоступна.',
  timeout: 'Облачная модель не ответила вовремя.',
  network_error: 'Прервалось соединение сервера с облачной моделью.',
  invalid_response: 'Облачная модель вернула неполный ответ.',
  vertex_error: 'Google Vertex AI вернул ошибку.',
  disabled: 'Облачный анализ выключен на сервере.',
};
export const cloudStatusText = status => cloudStatuses[status] || 'Сервер не получил пригодную облачную оценку.';

export class AnalysisUnavailable extends Error {
  constructor(message, {pauseBatch = false, rejectImage = false} = {}) {
    super(message);
    this.pauseBatch = pauseBatch;
    this.rejectImage = rejectImage;
  }
}

export async function loadEndpoint() {
  try {
    const response = await fetch(new URL('./hybrid-config.json', import.meta.url), {
      cache: 'no-store', credentials: 'omit', signal: AbortSignal.timeout(7000),
    });
    if (!response.ok) throw Error();
    const config = await response.json();
    const url = new URL(config.endpoint);
    const loopback = ['localhost','127.0.0.1','[::1]'].includes(url.hostname);
    if (url.username || url.password || (url.protocol !== 'https:' && !(loopback && url.protocol === 'http:'))) throw Error();
    return url.href;
  } catch {
    throw new AnalysisUnavailable('Адрес сервера анализа недоступен. Фото не отправлено.', {pauseBatch: true});
  }
}

async function jpegForUpload(bitmap) {
  const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext('2d', {alpha: false});
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  for (const quality of [.88, .68]) {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    if (blob?.type === 'image/jpeg' && blob.size > 0 && blob.size <= maxJpegBytes) return blob;
  }
  throw Error();
}

function usableResult(data) {
  if (!data || !['hybrid','server_local'].includes(data.mode) || !knownObjectTypes.includes(data.object_type)
    || !named(data.model_version) || !probability(data.object_type_score) || !validScores(data.scores)) return false;
  const local = data.analysis?.local, cloud = data.analysis?.cloud, weights = data.analysis?.weights;
  if (!named(local?.model) || !validScores(local.scores) || !named(cloud?.status)
    || !probability(weights?.local) || !probability(weights?.cloud) || Math.abs(weights.local + weights.cloud - 1) > .001) return false;
  if (data.mode === 'hybrid') return named(cloud.model) && cloud.status === 'ok' && validScores(cloud.scores) && weights.cloud > 0;
  return weights.cloud === 0 && weights.local === 1;
}

export async function analyzeOnServer(bitmap, endpoint, progress) {
  let jpeg;
  try { jpeg = await jpegForUpload(bitmap); }
  catch { throw new AnalysisUnavailable('Не удалось подготовить JPEG. Фото не отправлено.'); }
  progress('Отправляем JPEG. Сервер запускает локальную модель и запрашивает второе мнение Google Vertex AI…');
  const waiting = setTimeout(() => progress('Ожидаем ответ сервера. Первый запуск может занять несколько минут…'), 9000);
  try {
    const headers = {'Content-Type': 'image/jpeg'};
    if (new URL(endpoint).hostname.endsWith('.ngrok-free.dev')) headers['ngrok-skip-browser-warning'] = '1';
    const response = await fetch(endpoint, {
      method: 'POST', body: jpeg, headers,
      credentials: 'omit', redirect: 'error', referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(180000),
    });
    if (response.status === 429) throw new AnalysisUnavailable('Сервер достиг лимита запросов. Повторная отправка отключена.', {pauseBatch: true});
    if ([202,502,503,504].includes(response.status)) throw new AnalysisUnavailable('Сервер прогревается или временно недоступен. Повторная отправка отключена.', {pauseBatch: true});
    if ([401,403].includes(response.status)) throw new AnalysisUnavailable('Сервер отклонил доступ к полному анализу. Повторная отправка отключена.', {pauseBatch: true});
    if ([400,413,415,422].includes(response.status)) throw new AnalysisUnavailable('Сервер отклонил фотографию. Проверьте файл или выберите другой кадр.', {rejectImage: true});
    if (!response.ok) throw new AnalysisUnavailable('Сервер не вернул оценку. Повторная отправка отключена.', {pauseBatch: true});
    let data;
    try { data = await response.json(); } catch { /* Classified below with other incomplete responses. */ }
    if (!usableResult(data)) throw new AnalysisUnavailable('Получен неполный ответ сервера. Повторная отправка отключена.', {pauseBatch: true});
    return data;
  } catch (error) {
    if (error instanceof AnalysisUnavailable) throw error;
    const reason = error.name === 'TimeoutError' || error.name === 'AbortError'
      ? 'Время ожидания сервера истекло.' : 'Нет ответа от сервера.';
    throw new AnalysisUnavailable(reason + ' Доставку фото подтвердить не удалось. Повторная отправка отключена.', {pauseBatch: true});
  } finally { clearTimeout(waiting); }
}
