/**
 * Visual previews of the latest neutral letterbox and train augmentation policy.
 * Run: node report/generate_preprocessing_examples.js
 * This does not invoke Python, train models, or use the attached reference images.
 * The augmentation preview implements the recorded ranges in JavaScript for
 * illustration; it is not a saved torchvision minibatch from measured runs.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const assetDir = path.join(__dirname, 'report_assets', 'latest');
fs.mkdirSync(assetDir, {recursive: true});
let sharp;
try { sharp = require('sharp'); }
catch {
  const home = process.env.USERPROFILE || process.env.HOME || '';
  try { sharp = require(path.join(home, '.cache', 'codex-runtimes', 'codex-primary-runtime', 'dependencies', 'node', 'node_modules', 'sharp')); }
  catch { throw Error('Install the sharp Node package (npm install sharp) to generate these image previews.'); }
}
const N = 224;
const NEUTRAL = [124, 116, 104];
const classOrder = ['san_phraphrom', 'san_phraphum', 'san_tayay', 'tee_ju_eia'];
const classLabels = [['San Phra', 'Phrom'], ['San Phra', 'Phum'], ['San Ta', 'Yai'], ['Tee Ju', 'Eia']];
const classColors = require('./figure_theme').classColors;
function parseCsv(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) { if (c === '"' && text[i+1] === '"') { field += '"'; i++; } else if (c === '"') quoted = false; else field += c; }
    else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const header = rows.shift().map((v,i) => i === 0 ? v.replace(/^\uFEFF/, '') : v);
  return rows.filter(r => r.length === header.length).map(r => Object.fromEntries(header.map((h,i) => [h,r[i]])));
}
function hashFile(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
function seedFor(id) { return crypto.createHash('sha256').update('readme-augmentation-preview-v1|' + id).digest().readUInt32LE(0); }
function rng(seed) { let x = seed >>> 0; return () => { x = (x + 0x6D2B79F5) >>> 0; let z = Math.imul(x ^ (x >>> 15), 1 | x); z ^= z + Math.imul(z ^ (z >>> 7), 61 | z); return ((z ^ (z >>> 14)) >>> 0) / 4294967296; }; }
function between(rand, lo, hi) { return lo + rand() * (hi - lo); }
function clamp(v) { return Math.max(0, Math.min(255, Math.round(v))); }
function emptyCanvas() { const a = Buffer.alloc(N*N*3); for(let i=0;i<a.length;i+=3){a[i]=NEUTRAL[0];a[i+1]=NEUTRAL[1];a[i+2]=NEUTRAL[2];} return a; }
async function openRgb(file) {
  const {data,info} = await sharp(file, {animated:false,page:0}).rotate().flatten({background:{r:NEUTRAL[0],g:NEUTRAL[1],b:NEUTRAL[2]}}).toColourspace('srgb').removeAlpha().raw().toBuffer({resolveWithObject:true});
  if(info.channels !== 3) throw Error(`Expected RGB for ${file}, got ${info.channels} channels`);
  return {data,width:info.width,height:info.height};
}
async function letterbox(source) {
  const scale = Math.min(N/source.width, N/source.height);
  const width = Math.max(1, Math.round(source.width*scale)), height = Math.max(1,Math.round(source.height*scale));
  const resized = await sharp(source.data,{raw:{width:source.width,height:source.height,channels:3}}).resize(width,height,{kernel:'lanczos3'}).raw().toBuffer();
  const canvas=emptyCanvas(), left=Math.floor((N-width)/2), top=Math.floor((N-height)/2);
  for(let y=0;y<height;y++) resized.copy(canvas,((top+y)*N+left)*3,y*width*3,(y+1)*width*3);
  return {data:canvas,bbox:[left,top,left+width,top+height]};
}
function warp(input, mapper) {
  const out=emptyCanvas();
  for(let y=0;y<N;y++) for(let x=0;x<N;x++) {
    const [sx,sy]=mapper(x,y),ix=Math.round(sx),iy=Math.round(sy);
    if(ix<0||ix>=N||iy<0||iy>=N) continue;
    const from=(iy*N+ix)*3,to=(y*N+x)*3;
    out[to]=input[from];out[to+1]=input[from+1];out[to+2]=input[from+2];
  }
  return out;
}
function homography(dest, src) {
  const a=[];
  for(let i=0;i<4;i++) {
    const [x,y]=dest[i], [u,v]=src[i];
    a.push([x,y,1,0,0,0,-u*x,-u*y,u]);
    a.push([0,0,0,x,y,1,-v*x,-v*y,v]);
  }
  for(let col=0;col<8;col++) {
    let pivot=col;for(let r=col+1;r<8;r++)if(Math.abs(a[r][col])>Math.abs(a[pivot][col]))pivot=r;
    [a[col],a[pivot]]=[a[pivot],a[col]];
    if(Math.abs(a[col][col])<1e-10)throw Error('Degenerate perspective transform');
    const div=a[col][col];for(let c=col;c<9;c++)a[col][c]/=div;
    for(let r=0;r<8;r++)if(r!==col){const f=a[r][col];for(let c=col;c<9;c++)a[r][c]-=f*a[col][c];}
  }
  const h=a.map(r=>r[8]);
  return (x,y)=>{const d=h[6]*x+h[7]*y+1;return [(h[0]*x+h[1]*y+h[2])/d,(h[3]*x+h[4]*y+h[5])/d];};
}
function rgbToHsv(r,g,b){r/=255;g/=255;b/=255;const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;let h=0;if(d){if(max===r)h=((g-b)/d)%6;else if(max===g)h=(b-r)/d+2;else h=(r-g)/d+4;h/=6;if(h<0)h+=1;}return [h,max===0?0:d/max,max];}
function hsvToRgb(h,s,v){const a=v*s,f=n=>{const k=(n+h*6)%6;return v-a*Math.max(0,Math.min(k,4-k,1));};return [f(5),f(3),f(1)].map(z=>clamp(z*255));}
function jitter(input, factors){
  const out=Buffer.from(input);let avg=0;for(let i=0;i<out.length;i+=3)avg+=(out[i]+out[i+1]+out[i+2])/3;avg/=N*N;
  for(let i=0;i<out.length;i+=3){let rgb=[out[i],out[i+1],out[i+2]].map(v=>clamp((v-avg)*factors.contrast+avg));const gray=.299*rgb[0]+.587*rgb[1]+.114*rgb[2];rgb=rgb.map(v=>clamp(gray+(v-gray)*factors.saturation));const hsv=rgbToHsv(...rgb);hsv[0]=(hsv[0]+factors.hue+1)%1;rgb=hsvToRgb(...hsv).map(v=>clamp(v*factors.brightness));out[i]=rgb[0];out[i+1]=rgb[1];out[i+2]=rgb[2];}
  return out;
}
function augment(canvas, rand) {
  const p={rotation_deg:between(rand,-12,12),translate_x_px:0,translate_y_px:0,scale:between(rand,.92,1.08),perspective:false,horizontal_flip:false,brightness:between(rand,.9,1.1),contrast:between(rand,.9,1.1),saturation:between(rand,.9,1.1),hue:between(rand,-.02,.02),erasing:false};
  let a=canvas, cx=(N-1)/2, cy=(N-1)/2, t=p.rotation_deg*Math.PI/180, co=Math.cos(t), si=Math.sin(t);
  a=warp(a,(x,y)=>[cx+co*(x-cx)+si*(y-cy),cy-si*(x-cx)+co*(y-cy)]);
  p.translate_x_px=Math.round(between(rand,-.05*N,.05*N));p.translate_y_px=Math.round(between(rand,-.05*N,.05*N));
  a=warp(a,(x,y)=>[cx+(x-cx-p.translate_x_px)/p.scale,cy+(y-cy-p.translate_y_px)/p.scale]);
  if(rand()<.20){const corners=[[0,0],[N-1,0],[N-1,N-1],[0,N-1]],d=corners.map(([x,y])=>[x+between(rand,-.05*N,.05*N),y+between(rand,-.05*N,.05*N)]);a=warp(a,homography(d,corners));p.perspective=true;}
  if(rand()<.5){a=warp(a,(x,y)=>[N-1-x,y]);p.horizontal_flip=true;}
  a=jitter(a,p);
  if(rand()<.05){const area=between(rand,.02,.08)*N*N,ratio=Math.exp(between(rand,Math.log(.3),Math.log(3.3))),w=Math.min(N,Math.max(1,Math.round(Math.sqrt(area*ratio)))),h=Math.min(N,Math.max(1,Math.round(Math.sqrt(area/ratio)))),x0=Math.floor(rand()*(N-w+1)),y0=Math.floor(rand()*(N-h+1));for(let y=y0;y<y0+h;y++)for(let x=x0;x<x0+w;x++){const i=(y*N+x)*3;a[i]=NEUTRAL[0];a[i+1]=NEUTRAL[1];a[i+2]=NEUTRAL[2];}p.erasing=true;p.erase_bbox=[x0,y0,w,h];}
  return {data:a,params:p};
}
function escapeXml(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');}
async function png(raw,width=N,height=N){return sharp(raw,{raw:{width,height,channels:3}}).png().toBuffer();}
async function paddingPanel(row) {
  const sourceFile=path.join(root,'dataset_cleaned',row.path),source=await openRgb(sourceFile),letter=await letterbox(source);
  const original=await sharp(source.data,{raw:{width:source.width,height:source.height,channels:3}}).resize({width:720,height:560,fit:'inside',kernel:'lanczos3'}).png().toBuffer();
  const om=await sharp(original).metadata(),padded=await sharp(await png(letter.data)).resize(640,640,{kernel:'nearest'}).png().toBuffer();
  const W=1540,H=825,svg=Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><style>text{font-family:Arial,sans-serif;fill:#263238}</style><rect width="100%" height="100%" fill="white"/><text x="770" y="42" text-anchor="middle" font-size="27" font-weight="bold">Aspect-ratio-preserving neutral letterbox</text><text x="405" y="92" text-anchor="middle" font-size="22">Original (${source.width} × ${source.height})</text><text x="1150" y="92" text-anchor="middle" font-size="22">Model input (224 × 224; enlarged view)</text><text x="770" y="805" text-anchor="middle" font-size="16">${escapeXml(row.image_id)} · train split · neutral fill RGB (124,116,104)</text></svg>`);
  const panel=await sharp({create:{width:W,height:H,channels:3,background:'#fff'}}).composite([{input:svg,left:0,top:0},{input:original,left:405-Math.floor(om.width/2),top:125+Math.floor((640-om.height)/2)},{input:padded,left:830,top:125}]).png().toBuffer();
  fs.writeFileSync(path.join(assetDir,'padding_large.png'),panel);
  return {image_id:row.image_id,path:row.path,original_width:source.width,original_height:source.height,model_input_size:N,letterbox_bbox:letter.bbox,source_sha256:hashFile(sourceFile)};
}
async function augmentationGrid(rows) {
  const chosen=[];for(const label of classOrder){const candidates=rows.filter(r=>r.split==='train'&&r.label===label);const distinct=new Map();for(const r of candidates){const key=r.group_id||r.image_id;if(!distinct.has(key))distinct.set(key,r);}const pick=[...distinct.values()].sort((a,b)=>seedFor(a.image_id)-seedFor(b.image_id)).slice(0,5);if(pick.length!==5)throw Error(`Expected five independent train groups: ${label}`);chosen.push(...pick);}
  const W=1390,H=1120,tile=224,x0=190,y0=100,dx=235,dy=250,composites=[],manifest=[];
  for(let i=0;i<chosen.length;i++){const r=chosen[i],file=path.join(root,'dataset_cleaned',r.path),source=await openRgb(file),base=await letterbox(source),a=augment(base.data,rng(seedFor(r.image_id)));composites.push({input:await png(a.data),left:x0+(i%5)*dx,top:y0+Math.floor(i/5)*dy});manifest.push({class:r.label,image_id:r.image_id,path:r.path,split:r.split,group_id:r.group_id,source_sha256:hashFile(file),seed:seedFor(r.image_id),...a.params});}
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><style>text{font-family:Arial,sans-serif;fill:#263238}</style><rect width="100%" height="100%" fill="white"/><text x="${W/2}" y="43" text-anchor="middle" font-size="27" font-weight="bold">Training augmentation preview · 5 images per class</text>`;
  for(let c=0;c<5;c++)svg+=`<text x="${x0+c*dx+112}" y="88" text-anchor="middle" font-size="16">Example ${c+1}</text>`;
  for(let r=0;r<4;r++){const y=y0+r*dy+105;svg+=`<rect x="18" y="${y-57}" width="6" height="115" fill="${classColors[r]}"/><text x="34" y="${y-7}" font-size="18" font-weight="bold">${classLabels[r][0]}</text><text x="34" y="${y+17}" font-size="18" font-weight="bold">${classLabels[r][1]}</text>`;}
  svg+='</svg>';composites.unshift({input:Buffer.from(svg),left:0,top:0});
  const grid=await sharp({create:{width:W,height:H,channels:3,background:'#fff'}}).composite(composites).png().toBuffer();fs.writeFileSync(path.join(assetDir,'augmentation_train_5_per_class.png'),grid);
  return manifest;
}
async function main(){
  const splitFile=path.join(root,'outputs_modes_v3_td','splits.csv'),rows=parseCsv(fs.readFileSync(splitFile,'utf8'));
  const pad=rows.find(r=>r.image_id==='san_phraphrom/san_phraphrom_0001'&&r.split==='train');if(!pad)throw Error('Requested train example is not in the canonical train split');
  const padding=await paddingPanel(pad),aug=await augmentationGrid(rows);
  const manifest={created_utc:new Date().toISOString(),split_csv_sha256:hashFile(splitFile),source_policy:'shrine_classifier/data.py letterbox; shrine_classifier/training.py ImageDataset transforms',rendering_note:'Deterministic JavaScript illustration of the configured policy. Source images are actual train-split images; this is not a logged training minibatch or bit-identical torchvision RNG output.',padding,augmentation:aug};
  fs.writeFileSync(path.join(__dirname,'preprocessing_preview_sources.json'),JSON.stringify(manifest,null,2)+'\n');
  console.log(`Generated large padding example and ${aug.length} augmentation previews from canonical train images.`);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
