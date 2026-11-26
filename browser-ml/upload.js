import {assessInputQuality} from './input-quality.js';
import {recognize} from './infer.js';
import {openOrder, restoreOrder} from './workorder.js';
import {displayRoutes,existingEntry,saveEntry} from './route-batch.js';

const form=document.querySelector('#input'), status=document.querySelector('#status'), box=document.querySelector('#order');
const say=text=>{status.textContent=text;};
restoreOrder(box).catch(()=>say('Хранилище браузера сейчас недоступно.'));
displayRoutes();

async function processFile(file,route,progress){
  if(file.size>20*1024*1024)throw Error('Выберите файл размером до 20 МБ.');
  if(!['image/jpeg','image/png','image/webp'].includes(file.type))throw Error('Поддерживаются JPEG, PNG и WebP.');
  let hash='';
  if(route){
    const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    hash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
    if(existingEntry(route,hash))return {duplicate:true};
  }
  const bitmap=await createImageBitmap(file);
  try{
    if(bitmap.width*bitmap.height>40_000_000)throw Error('Выберите фотографию до 40 мегапикселей.');
    const quality=assessInputQuality(bitmap);
    if(!quality.ok)return {file:file.name,hash,reason:'Нужна пересъёмка. '+quality.request};
    const result=await recognize(bitmap,progress);
    const objectType=result.object_type==='telecom_cabinet'?'ТКШ':'РЩ';
    const orderId=await openOrder(box,{...result,object_type:objectType,filename:file.name,image:file,routeName:route});
    return {file:file.name,hash,orderId,objectType};
  }finally{bitmap.close();}
}

form.addEventListener('submit',async event=>{
  event.preventDefault();
  const button=document.querySelector('#analyze'), source=document.querySelector('#photo'), nameInput=document.querySelector('#route-name');
  const files=Array.from(source.files||[]);
  if(!files.length||button.disabled)return;
  let route=nameInput.value.trim();
  if(files.length>1 && !route){
    route='Выезд '+new Date().toLocaleString('ru-RU')+' · '+crypto.randomUUID().slice(0,4);
    nameInput.value=route;
  }
  button.disabled=true;source.disabled=true;nameInput.disabled=true;
  let ready=0,failed=0,duplicates=0,volatile=false;
  try{
    for(const [index,file] of files.entries()){
      const progress=text=>say((files.length>1?`Кадр ${index+1} из ${files.length}. `:'')+text);
      progress('Читаем фотографию…');
      const started=performance.now();
      let row;
      try {row=await processFile(file,route,progress);}
      catch(error){row={file:file.name,reason:'Не удалось завершить анализ. '+(error.message||'Проверьте файл и повторите.')};}
      if(row.duplicate){duplicates++;continue;}
      row.processingMs=Math.round(performance.now()-started);
      if(route && !saveEntry(route,row))volatile=true;
      if(row.orderId)ready++;else failed++;
      if(files.length===1)say(row.orderId?'Осмотр готов. Проверьте признаки в карточке.':row.reason);
    }
    if(route)displayRoutes(route,volatile);
    if(files.length>1||duplicates)say(`Обработка завершена. Новых карточек: ${ready}. На проверку файла или пересъёмку: ${failed}. Уже учтено в рейсе: ${duplicates}.`);
    if(files.length===1&&ready)box.scrollIntoView({block:'start'});
  }finally{button.disabled=false;source.disabled=false;nameInput.disabled=false;}
});
