import {recognize} from './infer.js';
import {openOrder, restoreOrder} from './workorder.js';
const form=document.querySelector('#input'), status=document.querySelector('#status'), box=document.querySelector('#order');
const say=t=>status.textContent=t;
restoreOrder(box).catch(()=>say('Хранилище браузера сейчас недоступно.'));
form.addEventListener('submit',async event=>{
  event.preventDefault();const button=document.querySelector('#analyze'),file=document.querySelector('#photo').files[0];if(!file||button.disabled)return;
  button.disabled=true;
  let bitmap;
  try{
    if(file.size>20*1024*1024)throw Error('Выберите файл размером до 20 МБ.');
    if(!['image/jpeg','image/png','image/webp'].includes(file.type))throw Error('Поддерживаются JPEG, PNG и WebP.');
    say('Читаем фотографию…');
    bitmap=await createImageBitmap(file);
    if(bitmap.width*bitmap.height>40_000_000)throw Error('Выберите фотографию до 40 мегапикселей.');
    const result=await recognize(bitmap,say);
    await openOrder(box,{...result,object_type:result.object_type==='telecom_cabinet'?'ТКШ':'РЩ',filename:file.name,image:file});
    say('Осмотр готов. Проверьте признаки в карточке.');box.scrollIntoView({block:'start'});
  }catch(e){say('Не удалось завершить анализ. '+(e.message||'Попробуйте ещё раз.'))}
  finally{bitmap?.close();button.disabled=false}
});
