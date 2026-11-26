import {condition,recommendation} from './order-state.js';
// Route metadata contains no images. Confirmed work and coordinates are read from the current card at export.
import {listOrders, showOrder} from './workorder.js?v=20261126-hybrid1';

const prefix = 'mgts-route-report-v1/';
const temporary = new Map();
const columns = ['Рейс','Файл','Карточка','ID оборудования','Тип','Состояние по фото','Рекомендация','Действия мастера','Приёмка','Широта шкафа','Долгота шкафа','Источник положения','МП','ЛКП','Обработка кадра, мс'];
const labels = {mechanical:'МП',coating:'ЛКП'};

function read(name) {
  try {
    const data=localStorage.getItem(prefix+name);
    if (data) return JSON.parse(data);
  } catch { /* The in-memory route remains available. */ }
  return temporary.get(name) || [];
}
function names() {
  const all=new Set(temporary.keys());
  try {
    for(let i=0;i<localStorage.length;i++) {
      const key=localStorage.key(i);
      if(key?.startsWith(prefix))all.add(key.slice(prefix.length));
    }
  } catch { /* Storage may be disabled. */ }
  return [...all].sort((a,b)=>a.localeCompare(b,'ru'));
}
export function existingEntry(name, hash) {
  return read(name).find(row => row.hash === hash && row.orderId);
}
export function saveEntry(name, row) {
  const rows=read(name).filter(old => !row.hash || old.hash !== row.hash);
  rows.push(row);
  temporary.set(name,rows);
  try { localStorage.setItem(prefix+name,JSON.stringify(rows));return true; } catch {return false;}
}
function csvCell(value) {
  let text=String(value ?? '');
  if (/^[\s\uFEFF]*[=+@-]/.test(text) && typeof value!=='number') text="'"+text;
  return '"'+text.replaceAll('"','""')+'"';
}
function detailedRow(name,row,card) {
  if (!card) return [name,row.file,row.orderId||'','','','Недостаточно данных',row.reason || 'Открыть карточку в браузере, где проходил осмотр','','','','','Не подтверждено','','',row.processingMs ?? ''];
  const confirmed=[],accepted=[];
  for(const [key,state] of Object.entries(card.signals || {})) {
    if (state.decision==='confirmed') confirmed.push(labels[key]);
    if (state.result) accepted.push(labels[key]+' · '+(state.result==='resolved'?'принято':'оставлено открытым'));
  }
  return [name,row.file,card.id,card.assetId||'',card.object_type,condition(card),recommendation(card),confirmed.join(', '),accepted.join(', '),
    card.location?.lat ?? '',card.location?.lon ?? '',card.location?'Введено мастером, WGS84':'Не подтверждено',
    card.scores?.mechanical ?? '',card.scores?.coating ?? '',row.processingMs ?? ''];
}
async function download(name) {
  const cards=new Map((await listOrders()).map(c=>[c.id,c]));
  const rows=[columns,...read(name).map(row=>detailedRow(name,row,cards.get(row.orderId)))];
  const text='\uFEFF'+rows.map(row=>row.map(csvCell).join(';')).join('\r\n')+'\r\n';
  const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='Рейс-'+name.replace(/[^\p{L}\p{N} _-]+/gu,'_').slice(0,60)+'.csv';
  document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
function element(tag,text) {
  const el=document.createElement(tag);
  if(text)el.textContent=text;
  return el;
}
export function displayRoutes(name='',volatile=false) {
  const host=document.querySelector('#route-results'), input=document.querySelector('#route-name');
  const routes=names();
  const options=document.querySelector('#route-names');
  options.replaceChildren(...routes.map(n=>{const opt=element('option');opt.value=n;return opt;}));
  host.hidden=!routes.length;
  if(!routes.length)return;
  const selected=routes.includes(name)?name:routes.includes(input.value)?input.value:routes.at(-1);
  const entries=read(selected), ready=entries.filter(r=>r.orderId).length;
  const title=element('h2','Итоги рейса');
  const label=element('label','Рейс для просмотра ');
  const select=element('select');
  for(const n of routes){const opt=element('option',n);opt.selected=n===selected;select.append(opt);}
  select.onchange=()=>{input.value=select.value;displayRoutes(select.value);};label.append(select);
  const summary=element('p',`Кадров в отчёте: ${entries.length}. Карточек: ${ready}. Требуют повторной съёмки или проверки файла: ${entries.length-ready}.`);
  const exportButton=element('button','Скачать реестр рейса (CSV)');exportButton.type='button';exportButton.className='wo-button';exportButton.onclick=()=>download(selected);
  const details=element('details'),list=element('ul');details.append(element('summary','Кадры и результаты'));
  for(const row of entries){
    const item=element('li');
    if(row.orderId) {
      const button=element('button',row.file+' · '+row.objectType);button.type='button';button.className='wo-button';
      button.onclick=async()=>{const dest=document.querySelector('#order');if(await showOrder(dest,row.orderId))dest.scrollIntoView({block:'start'});else summary.textContent='Карточка недоступна в этом хранилище браузера.';};item.append(button);
    }else item.textContent=row.file+' · '+row.reason;
    list.append(item);
  }
  details.append(list);
  host.replaceChildren(title,label,summary,exportButton,details);
  if(volatile)host.append(element('p','Этот рейс доступен в текущей вкладке. Скачайте CSV перед закрытием.'));
}
