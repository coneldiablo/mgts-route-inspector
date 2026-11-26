import * as ort from './runtime/ort.wasm.min.mjs';

const size = 224;
const mean = [.485,.456,.406], std = [.229,.224,.225];
let resources;

async function initialize() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('./runtime/', import.meta.url).href;
  const [session, recipe] = await Promise.all([
    ort.InferenceSession.create(new URL('./runtime/dino-int8.onnx',import.meta.url).href, {executionProviders:['wasm']}),
    fetch(new URL('./heads-v2.json',import.meta.url)).then(r => {if (!r.ok) throw Error('Не удалось загрузить параметры модели.');return r.json()})
  ]);
  return {session, recipe};
}

function sample(pixels, width, height, bounds) {
  const [left, top, right, bottom] = bounds;
  const out = new Float32Array(3*size*size);
  const axis = (n, start, end) => Array.from({length:size*4}, (_,i) =>
    Math.min(n-1, Math.floor(n*(start+(i+.5)/(size*4)*(end-start)))));
  const xs = axis(width,left,right), ys = axis(height,top,bottom);
  for (let y=0; y<size; y++) {
    for (let x=0; x<size; x++) {
      let red=0, green=0, blue=0;
      for (let j=0; j<4; j++) {
        const offset=ys[y*4+j]*width;
        for (let k=0; k<4; k++) {
          const p=4*(offset+xs[x*4+k]);
          red+=pixels[p]; green+=pixels[p+1]; blue+=pixels[p+2];
        }
      }
      const index=y*size+x;
      out[index]=(red/(16*255)-mean[0])/std[0];
      out[size*size+index]=(green/(16*255)-mean[1])/std[1];
      out[2*size*size+index]=(blue/(16*255)-mean[2])/std[2];
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
  const pix=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const features=new Float64Array(recipe.heads.telecom.weights.length);
  const viewFeatures=[];
  for (const [i, bounds] of recipe.views.entries()) {
    status(`Проверяем области кадра · ${i+1} из ${recipe.views.length}`);
    await new Promise(requestAnimationFrame);
    const input=sample(pix,bitmap.width,bitmap.height,bounds);
    let output;
    try {
      output=(await session.run({pixel_values:input})).last_hidden_state;
      const vector=embedTokens(output);
      viewFeatures.push(vector);
      for (let j=0;j<features.length;j++) features[j]+=vector[j]/recipe.views.length;
    } finally {
      input.dispose();
      output?.dispose();
    }
  }
  const scores={}, regions={};
  for (const [code, {weights,bias}] of Object.entries(recipe.heads)) {
    let value=bias;for(let i=0;i<features.length;i++) value+=features[i]*weights[i];
    scores[code]=1/(1+Math.exp(-value));
    if (code !== 'telecom') {
      // Each window contributes its linear score equally to the final logit.
      const contributions=viewFeatures.map(v=>v.reduce((sum,x,j)=>sum+weights[j]*x,0));
      const highest=Math.max(...contributions);
      if (highest-Math.min(...contributions)>1e-8) {
        const view=contributions.indexOf(highest);
        regions[code]={view, bounds:recipe.views[view]};
      }
    }
  }
  return {object_type:scores.telecom>=.5?'telecom_cabinet':'distribution_board',
          regions,model_version:'DINOv2 INT8 · 4 фрагмента',
          scores:{mechanical:scores.mechanical,coating:scores.coating},object_type_score:Math.max(scores.telecom,1-scores.telecom)};
}
