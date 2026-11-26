import * as ort from './runtime/ort.wasm.min.mjs';

const size = 224;
const mean = [.485,.456,.406], std = [.229,.224,.225];
let resources;

async function initialize() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('./runtime/', import.meta.url).href;
  const [session, recipe] = await Promise.all([
    ort.InferenceSession.create(new URL('./runtime/dino-int8.onnx',import.meta.url).href, {executionProviders:['wasm']}),
    fetch(new URL('./heads.json',import.meta.url)).then(r => {if (!r.ok) throw Error('Не удалось загрузить параметры модели.');return r.json()})
  ]);
  return {session, recipe};
}

function sample(pixels, width, height, inset) {
  const out = new Float32Array(3*size*size);
  const axis = n => Array.from({length:size*4}, (_,i) => Math.min(n-1,Math.floor(n*(inset+(i+.5)/(size*4)*(1-2*inset)))));
  const xs = axis(width), ys = axis(height);
  for (let y=0; y<size; y++) {
    for (let x=0; x<size; x++) {
      const rgb=[0,0,0];
      for (let j=0; j<4; j++) {
        const offset=ys[y*4+j]*width;
        for (let k=0; k<4; k++) {
          const p=4*(offset+xs[x*4+k]);
          for(let c=0;c<3;c++) rgb[c]+=pixels[p+c];
        }
      }
      for(let c=0;c<3;c++) out[c*size*size+y*size+x]=(rgb[c]/(16*255)-mean[c])/std[c];
    }
  }
  return new ort.Tensor('float32',out,[1,3,size,size]);
}

function embedTokens(tensor) {
  const [,count,dim]=tensor.dims, tokens=tensor.data;
  const cls=tokens.slice(0,dim), patches=new Float32Array(dim);
  for(let t=1;t<count;t++) for(let i=0;i<dim;i++) patches[i]+=tokens[t*dim+i]/(count-1);
  const unit = vec => {const norm=Math.hypot(...vec)||1;return Array.from(vec,v=>v/norm)};
  return [...unit(cls),...unit(patches)];
}

export async function recognize(bitmap, status) {
  status('Загрузка модели при первом запуске · около 36 МБ');
  if (!resources) resources=initialize().catch(e=>{resources=null;throw e});
  const {session,recipe}=await resources;
  const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(bitmap,0,0);
  const pix=ctx.getImageData(0,0,canvas.width,canvas.height).data, features=[];
  for (const [i, inset] of [0,.15].entries()) {
    status(i?'Проверяем центральную часть шкафа…':'Проверяем общий вид шкафа…');
    await new Promise(requestAnimationFrame);
    const input=sample(pix,bitmap.width,bitmap.height,inset);
    const outputs=await session.run({pixel_values:input});
    features.push(...embedTokens(outputs.last_hidden_state));
    input.dispose();outputs.last_hidden_state.dispose();
  }
  const scores={};
  for (const [code, {weights,bias}] of Object.entries(recipe.heads)) {
    let value=bias;for(let i=0;i<features.length;i++) value+=features[i]*weights[i];
    scores[code]=1/(1+Math.exp(-value));
  }
  return {object_type:scores.telecom>=.5?'telecom_cabinet':'distribution_board',
          scores:{mechanical:scores.mechanical,coating:scores.coating},object_type_score:Math.max(scores.telecom,1-scores.telecom)};
}
