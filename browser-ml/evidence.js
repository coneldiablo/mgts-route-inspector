import {modeNames} from './hybrid.js?v=20261126-hybrid1';
import {condition,recommendation} from './order-state.js';
// Legacy cards predate evidence identifiers. Their latest post-repair image is the only possible receipt.
export function evidenceId(entry) {
  if (!entry) return null;
  return entry.id || ('legacy:' + entry.orderId + ':' + entry.filename);
}

export function allAfter(order) {
  return [...(order.afterHistory || []), ...(order.after ? [order.after] : [])];
}

export function receiptFor(order, signal) {
  const state=order.signals[signal];
  if (!state.result) return null;
  return state.evidenceId
    ? allAfter(order).find(e=>evidenceId(e)===state.evidenceId)
    : order.after;
}

export function previousRetakes(order) {
  const entries=[...(order.retakeHistory || [])];
  if (order.request) entries.push({request:order.request,evidence:order.rephoto});
  return entries;
}

export function strongestRegion(order,signal) {
  const b=order.regions?.[signal]?.bounds;
  if (!Array.isArray(b)||b.length!==4||!b.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||b[0]>=b[2]||b[1]>=b[3]) return null;
  return b;
}

export function regionName(bounds) {
  const top=(bounds[1]+bounds[3])/2<.5;
  const left=(bounds[0]+bounds[2])/2<.5;
  return top ? (left?'верхняя левая часть':'верхняя правая часть') : (left?'нижняя левая часть':'нижняя правая часть');
}

export function croppedCanvas(image,bounds) {
  const [x0,y0,x1,y1]=bounds,w=image.naturalWidth,h=image.naturalHeight;
  const sw=Math.max(1,(x1-x0)*w),sh=Math.max(1,(y1-y0)*h),scale=Math.min(1,900/sw,900/sh);
  const canvas=document.createElement('canvas'); canvas.width=Math.max(1,Math.round(sw*scale));canvas.height=Math.max(1,Math.round(sh*scale));
  canvas.getContext('2d').drawImage(image,x0*w,y0*h,sw,sh,0,0,canvas.width,canvas.height);
  return canvas;
}

export async function downloadReport(order,signals,views) {
  const report=document.implementation.createHTMLDocument('Акт осмотра '+order.id);
  report.documentElement.lang='ru';
  const append=(parent,tag,text)=>{const el=report.createElement(tag);if(text!=null)el.textContent=text;parent.append(el);return el};
  const style=append(report.head,'style');
  style.textContent='body{max-width:850px;margin:32px auto;padding:0 18px;color:#17334e;font:15px/1.5 system-ui,sans-serif}h1,h2{color:#134074}section,figure{break-inside:avoid;margin:16px 0;padding:14px;border:1px solid #bbcbd7}img{display:block;max-width:100%;max-height:320px}figcaption{margin:6px 0}p{margin:.35em 0}@media print{body{padding:0;margin:0}img{max-height:200px}}';
  const body=report.body,dt=new Intl.DateTimeFormat('ru',{dateStyle:'short',timeStyle:'short'});
  append(body,'h1','Акт осмотра '+order.id);
  append(body,'p',order.object_type+' · Осмотр от '+dt.format(order.createdAt));
  append(body,'p','Состояние по фото: '+condition(order));
  append(body,'p','Рекомендация: '+recommendation(order));
  if(order.routeName)append(body,'p','Рейс: '+order.routeName);
  if(order.assetId)append(body,'p','ID оборудования: '+order.assetId);
  if(order.mode)append(body,'p',modeNames[order.mode]);
  if(order.model_version)append(body,'p','Модель: '+order.model_version);
  if(order.analysis) {
    append(body,'p','Локальная модель сервера · '+order.analysis.local.model);
    if(order.analysis.cloud.model)append(body,'p',(order.mode==='hybrid'?'Ответ Google Vertex AI · ':'Запрошенная облачная модель · ')+order.analysis.cloud.model);
    if(order.mode==='hybrid')append(body,'p','Вклад локальной и облачной моделей · '+Math.round(order.analysis.weights.local*100)+'% / '+Math.round(order.analysis.weights.cloud*100)+'%');
  }
  if(order.modeDetail)append(body,'p',order.modeDetail);
  append(body,'p',order.location?'Положение объекта (WGS84, введено мастером): '+order.location.lat+', '+order.location.lon:'Положение объекта не указано.');
  const figures=new Map();
  async function photo(e,caption) {
    if (!e?.image) return;
    const key=evidenceId(e) || e.filename;
    if(figures.has(key))return;
    figures.set(key,true);
    const fig=append(body,'figure');append(fig,'figcaption',caption+' · '+e.filename);
    const img=append(fig,'img');img.alt=caption;
    const bitmap=await createImageBitmap(e.image);
    try {
      const ratio=Math.min(1,1000/Math.max(bitmap.width,bitmap.height)),canvas=document.createElement('canvas');
      canvas.width=Math.max(1,Math.round(bitmap.width*ratio));canvas.height=Math.max(1,Math.round(bitmap.height*ratio));
      canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);
      img.src=canvas.toDataURL('image/jpeg',.84);
    } finally {bitmap.close()}
  }
  await photo(order.before,'Исходный кадр');
  append(body,'h2','Работы по признакам');
  for(const [key,sig] of Object.entries(signals)){
    const state=order.signals[key],sec=append(body,'section');append(sec,'h3',sig.title);
    if(Number.isFinite(order.scores[key]))append(sec,'p','Оценка модели: '+(100*order.scores[key]).toFixed(1).replace('.',',')+'%');
    if(order.analysis) {
      const pct=value=>new Intl.NumberFormat('ru-RU',{style:'percent',maximumFractionDigits:1}).format(value);
      append(sec,'p','Локальная оценка · '+pct(order.analysis.local.scores[key]));
      append(sec,'p',order.mode==='hybrid'?'Облачная оценка · '+pct(order.analysis.cloud.scores[key]):'Облачная оценка отсутствует');
    }
    const region=strongestRegion(order,key);if(region)append(sec,'p','Наибольший вклад в оценку: '+regionName(region)+' исходного кадра.');
    append(sec,'p',sig.task);
    append(sec,'p','Решение мастера: '+({pending:'ожидает проверки',confirmed:'работа подтверждена',dismissed:'признак снят'}[state.decision]));
    const receipt=receiptFor(order,key);
    if(state.result)append(sec,'p','Приёмка: '+(state.result==='resolved'?'устранено':'оставлено в работе')+(receipt?' · '+receipt.filename:'')+(state.reviewedAt?' · '+dt.format(state.reviewedAt):''));
  }
  const requests=previousRetakes(order);
  if(requests.length){append(body,'h2','Пересъёмки');for(const {request,evidence} of requests){
    append(body,'p',signals[request.signal].code+' · '+views[request.view]+' · '+(request.source==='model'?'запрос системы':'запрос мастера')+' · '+(evidence?'получен кадр':'ожидается кадр'));
    await photo(evidence,'Пересъёмка · '+signals[request.signal].code);
  }}
  const after=allAfter(order);if(after.length){append(body,'h2','Фотографии после ремонта');for(const e of after){
    await photo(e,'После ремонта · '+(signals[e.scope]?.code || 'общий кадр'));
  }}
  const blob=new Blob(['<!doctype html>\n',report.documentElement.outerHTML],{type:'text/html;charset=utf-8'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=order.id+'-акт.html';
  document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
