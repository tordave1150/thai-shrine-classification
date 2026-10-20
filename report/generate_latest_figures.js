/** Rebuild README figures from the completed outputs_modes_v3_td experiment.
 * Run: node report/generate_latest_figures.js
 * Only reads existing audit, tuning, final, and confidence records; no model code runs.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const theme = require('./figure_theme');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'outputs_modes_v3_td');
const assets = path.join(__dirname, 'report_assets', 'latest');
fs.mkdirSync(assets, {recursive: true});
const archs = ['resnet50','efficientnet_b0','mobilenet_v3_large','densenet121','convnext_tiny'];
const names = {resnet50:'ResNet50',efficientnet_b0:'EfficientNet-B0',mobilenet_v3_large:'MobileNetV3-Large',densenet121:'DenseNet121',convnext_tiny:'ConvNeXt-Tiny'};
const modes = ['frozen','partial','full'];
const colors = theme.modeColors;
const symbols = theme.symbols;
const classes = ['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia'];
const classNames = ['San Phra Phrom','San Phra Phum','San Ta Yai','Tee Ju Eia'];
const shortClassNames = ['พระพรหม','พระภูมิ','ตายาย','ตี่จู้เอี๊ยะ'];
const classColors = theme.classColors;
const source = {};
read('report/figure_theme.js');
const winners = JSON.parse(read('outputs_modes_v3_td/tuning/best_params_by_arch_mode.json')).pairs;
function esc(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');}
function csv(s){let rows=[],row=[],field='',q=false;for(let i=0;i<s.length;i++){const c=s[i];if(q){if(c==='"'&&s[i+1]==='"'){field+='"';i++;}else if(c==='"')q=false;else field+=c;}else if(c==='"')q=true;else if(c===','){row.push(field);field='';}else if(c==='\n'){row.push(field.replace(/\r$/,''));rows.push(row);row=[];field='';}else field+=c;}if(field||row.length){row.push(field.replace(/\r$/,''));rows.push(row);}const h=rows.shift();return rows.filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])));}
function read(rel){const p=path.join(root,rel);const b=fs.readFileSync(p);source[rel]=crypto.createHash('sha256').update(b).digest('hex');return b.toString('utf8');}
function readCsv(rel){return csv(read(rel).replace(/^\uFEFF/,''));}
function save(name,svg){fs.writeFileSync(path.join(assets,name),svg+'\n');}
function fmt(n,d=1){return Number(n).toFixed(d);}
function mean(a){return a.reduce((x,y)=>x+y,0)/a.length;}
function sd(a){const m=mean(a);return Math.sqrt(a.reduce((x,y)=>x+(y-m)**2,0)/(a.length-1));}
function quant(a,p){const b=[...a].sort((x,y)=>x-y),i=(b.length-1)*p,j=Math.floor(i);return b[j]+(b[Math.ceil(i)]-b[j])*(i-j);}
function svgStart(w,h,title){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><rect width="${w}" height="${h}" fill="white"/><style>${theme.styles}</style><text x="${w/2}" y="34" class="title" text-anchor="middle">${esc(title)}</text>`;}
function line(x1,y1,x2,y2,cls='axis',extra=''){const stroke=extra.includes('stroke="')?'':`stroke="${theme.muted}"`;return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}" ${stroke} ${extra}/>`;}
function txt(x,y,t,cls='label',anchor='middle',extra=''){return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${extra}>${esc(t)}</text>`;}
function rect(x,y,w,h,fill,extra=''){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" ${extra}/>`;}
function marker(x,y,mode,r=4,opacity=.75){const c=colors[mode];if(symbols[mode]==='square')return rect(x-r,y-r,2*r,2*r,c,`opacity="${opacity}"`);if(symbols[mode]==='diamond')return `<path d="M ${x} ${y-r*1.2} L ${x+r*1.2} ${y} L ${x} ${y+r*1.2} L ${x-r*1.2} ${y} Z" fill="${c}" opacity="${opacity}"/>`;return `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" opacity="${opacity}"/>`;}
function legend(x,y){let s='';modes.forEach((m,i)=>{const px=x+i*130;s+=marker(px,y-4,m,5,1)+txt(px+12,y,m[0].toUpperCase()+m.slice(1),'label','start');});return s;}
function box(x,values,map,color,width=23,mode=null){const lo=Math.min(...values),hi=Math.max(...values),q1=quant(values,.25),q2=quant(values,.5),q3=quant(values,.75);let s=line(x,map(lo),x,map(hi),'axis',`stroke="${color}"`)+line(x-8,map(lo),x+8,map(lo),'axis',`stroke="${color}"`)+line(x-8,map(hi),x+8,map(hi),'axis',`stroke="${color}"`);s+=rect(x-width/2,map(q3),width,Math.max(1,map(q1)-map(q3)),color,'opacity="0.55" stroke="#35434a"')+line(x-width/2,map(q2),x+width/2,map(q2),'axis','stroke="#263238" stroke-width="2"');if(values.length<=10)values.forEach((v,i)=>{const px=x+(i-(values.length-1)/2)*3.4;s+=mode?marker(px,map(v),mode,2.6,1):`<circle cx="${px}" cy="${map(v)}" r="2.5" fill="${color}" stroke="white" stroke-width=".5"/>`;});return s;}
function axes(x,y,w,h,min,max,ticks,label){let s=line(x,y+h,x+w,y+h)+line(x,y,x,y+h);ticks.forEach(v=>{const py=y+h-(v-min)/(max-min)*h;s+=line(x,py,x+w,py,'grid')+txt(x-8,py+4,fmt(v,Number.isInteger(v)?0:1),'small','end');});s+=txt(x+w/2,y+h+39,label,'label');return s;}

const audit = readCsv('outputs_modes_v3_td/dataset_audit/image_properties.csv');
if(audit.length!==560)throw Error(`Expected 560 audited images, got ${audit.length}`);
const finals=readCsv('outputs_modes_v3_td/final/run_metrics.csv').filter(r=>r.status==='completed');
if(finals.length!==75)throw Error(`Expected 75 completed final runs, got ${finals.length}`);
const trials=readCsv('outputs_modes_v3_td/tuning/all_trials.csv').filter(r=>r.status==='completed');
if(trials.length!==75)throw Error(`Expected 75 completed tuning trials, got ${trials.length}`);
const pairs=new Map();for(const r of finals){const k=`${r.architecture}/${r.mode}`;if(!pairs.has(k))pairs.set(k,[]);pairs.get(k).push(r);}for(const [k,v] of pairs)if(v.length!==5||new Set(v.map(r=>r.seed)).size!==5)throw Error(`Incomplete seed set for ${k}`);

// Dataset charts use the audited images, never the example screenshot counts.
{
 const counts=classes.map(c=>audit.filter(r=>r.label===c).length);let s=svgStart(920,430,'Cleaned dataset: images per class')+axes(82,77,790,285,0,220,[0,50,100,150,200],'Class');
 counts.forEach((v,i)=>{const x=110+i*196;s+=rect(x,77+285-v/220*285,123,v/220*285,classColors[i])+txt(x+61.5,72+285-v/220*285,v,'sub')+txt(x+61.5,388,classNames[i],'label');});save('class_distribution.svg',s+'</svg>');
}
{
 let s=svgStart(1200,510,'Image dimensions and aspect ratio by class');const left={x:80,y:75,w:510,h:360},right={x:690,y:75,w:450,h:360};
 s+=txt(330,67,'Original width vs. height (px)','sub')+txt(915,67,'Width / height (count)','sub');s+=axes(left.x,left.y,left.w,left.h,0,4500,[0,1000,2000,3000,4000],'Width (px)');[0,1000,2000,3000,4000].forEach(v=>s+=txt(left.x+v/4500*left.w,left.y+left.h+17,v,'small'));s+=line(left.x,left.y+left.h,left.x+left.w,left.y,'grid','stroke-dasharray="6 5"');
 audit.forEach(r=>{const i=classes.indexOf(r.label),x=left.x+Number(r.width)/4500*left.w,y=left.y+left.h-Number(r.height)/4500*left.h;s+=`<circle cx="${x}" cy="${y}" r="2.7" fill="${classColors[i]}" opacity=".58"/>`;});
 s+=txt(23,260,'Height (px)','label','middle','transform="rotate(-90 23 260)"');s+=axes(right.x,right.y,right.w,right.h,0,100,[0,20,40,60,80,100],'Aspect ratio (width / height)');
 const bins=Array.from({length:18},()=>[0,0,0,0]);audit.forEach(r=>{const v=Number(r.width)/Number(r.height),i=Math.min(17,Math.max(0,Math.floor((v-.3)/.1)));bins[i][classes.indexOf(r.label)]++;});
 bins.forEach((b,i)=>b.forEach((n,c)=>{const bx=right.x+i*(right.w/18)+c*5.5;s+=rect(bx,right.y+right.h-n/100*right.h,5,n/100*right.h,classColors[c],`opacity=".8"`);}));
 s+=txt(right.x,453,'0.3','small')+txt(right.x+right.w*.39,453,'1.0','small')+txt(right.x+right.w,453,'2.1+','small');classes.forEach((_,i)=>{s+=rect(80+i*235,482,12,12,classColors[i])+txt(98+i*235,493,classNames[i],'small','start');});save('image_dimensions.svg',s+'</svg>');
}
{
 let s=svgStart(1180,430,'Decoded colour modes in the latest audit'),ms=['RGB','RGBA','P','CMYK'];const total=ms.map(m=>audit.filter(r=>r.mode===m).length);s+=txt(274,67,'Overall count','sub')+txt(825,67,'Count by class','sub');s+=axes(70,83,460,275,0,560,[0,100,200,300,400,500],'Colour mode');
 ms.forEach((m,i)=>{const h=total[i]/560*275,x=95+i*107;s+=rect(x,358-h,67,h,['#374151','#6B7280','#9CA3AF','#D1D5DB'][i])+txt(x+33,351-h,total[i],'small')+txt(x+33,382,m,'label');});
 s+=axes(640,83,490,275,0,210,[0,50,100,150,200],'Class');classes.forEach((c,i)=>{ms.forEach((m,j)=>{const n=audit.filter(r=>r.label===c&&r.mode===m).length,h=n/210*275;s+=rect(660+i*119+j*19,358-h,17,h,['#374151','#6B7280','#9CA3AF','#D1D5DB'][j]);});s+=txt(690+i*119,380,classNames[i].replace('San ','').replace('Tee Ju ','TJ '),'small');});
 ms.forEach((m,i)=>{s+=rect(720+i*92,411,11,11,['#374151','#6B7280','#9CA3AF','#D1D5DB'][i])+txt(736+i*92,421,m,'small','start');});save('colour_modes.svg',s+'</svg>');
}

// Search response: saved nested trial configs, with exactly the two tuned numeric variables.
{
 let s=svgStart(1220,545,'Tuning trials: validation Macro F1 (%)');const panels=[{key:'learning_rate',label:'Learning rate (log scale)',min:1e-5,max:1e-3,x:80},{key:'weight_decay',label:'Weight decay (log scale)',min:1e-6,max:1e-2,x:665}];
 for(const p of panels){s+=txt(p.x+245,78,p.label,'sub')+axes(p.x,95,500,325,40,100,[40,50,60,70,80,90,100],p.label);for(let power=Math.log10(p.min);power<=Math.log10(p.max);power++)s+=txt(p.x+(power-Math.log10(p.min))/(Math.log10(p.max)-Math.log10(p.min))*500,440,`10^${power}`,'small');for(const r of trials){const hit=r.config.match(new RegExp("'"+p.key+"': ([0-9.eE+-]+)"));if(!hit)throw Error(`Missing ${p.key} in ${r.identity}`);const v=Number(hit[1]),f=Number(r.best_val_macro_f1)*100;if(!Number.isFinite(v)||!Number.isFinite(f))throw Error('Non-finite trial');const x=p.x+(Math.log10(v)-Math.log10(p.min))/(Math.log10(p.max)-Math.log10(p.min))*500,y=420-(f-40)/60*325;s+=marker(x,y,r.mode,4.2,.62);if(Number(r.trial)===Number(winners[r.architecture+"::"+r.mode]?.winning_trial))s+="<circle cx=\""+x+"\" cy=\""+y+"\" r=\"7\" fill=\"none\" stroke=\""+theme.ink+"\" stroke-width=\"1.3\"/>";}}s+=legend(405,515);save('tuning_response.svg',s+'</svg>');
}

// Four aligned panels: each dot is one of five final seeds, bars are mean ± sample SD.
{
 const W=1300,H=950;let s=svgStart(W,H,'Independently tuned modes: five matched final seeds')+legend(450,66);const metrics=[['best_val_acc','Validation accuracy (%)'],['test_acc','Test accuracy (%)'],['best_val_macro_f1','Validation Macro F1 (%)'],['test_macro_f1','Test Macro F1 (%)']];
 metrics.forEach(([key,label],pi)=>{const col=pi%2,row=Math.floor(pi/2),x=68+col*630,y=125+row*425,w=550,h=292;s+=txt(x+w/2,y-20,label,'sub')+axes(x,y,w,h,75,100,[75,80,85,90,95,100],'Architecture');archs.forEach((a,ai)=>{modes.forEach((m,mi)=>{const v=pairs.get(`${a}/${m}`).map(r=>Number(r[key])*100),cx=x+55+ai*105+(mi-1)*22,mu=mean(v),sigma=sd(v),fy=t=>y+h-(t-75)/25*h;s+=line(cx,fy(mu-sigma),cx,fy(mu+sigma),'axis',`stroke="${colors[m]}" stroke-width="2"`)+line(cx-6,fy(mu-sigma),cx+6,fy(mu-sigma),'axis',`stroke="${colors[m]}"`)+line(cx-6,fy(mu+sigma),cx+6,fy(mu+sigma),'axis',`stroke="${colors[m]}"`);v.forEach((z,i)=>s+=marker(cx+(i-2)*1.8,fy(z),m,2.7,.42));s+=marker(cx,fy(mu),m,5.5,1);});s+=txt(x+55+ai*105,y+h+18,names[a].replace('EfficientNet-','EffNet-').replace('MobileNetV3-','MobV3-').replace('ConvNeXt-','ConvNeXt-'),'small');});});save('mode_comparison.svg',s+'</svg>');
}

// Checkpoint box plots replace crowded temporal aggregates in README.
for(const a of archs){let s=svgStart(970,480,`${names[a]}: selected-checkpoint distributions (n = 5 seeds)`)+legend(286,69);const panels=[{key:'best_val_macro_f1',title:'Validation Macro F1 (%)',min:75,max:100,ticks:[75,80,85,90,95,100],scale:100},{key:'best_epoch',title:'Best epoch',min:0,max:30,ticks:[0,5,10,15,20,25,30],scale:1}];panels.forEach((p,pi)=>{const x=75+pi*470,y=115,w=395,h=265;s+=txt(x+w/2,y-16,p.title,'sub')+axes(x,y,w,h,p.min,p.max,p.ticks,'Fine-tuning mode');modes.forEach((m,i)=>{const v=pairs.get(`${a}/${m}`).map(r=>Number(r[p.key])*p.scale),cx=x+92+i*108,fy=z=>y+h-(z-p.min)/(p.max-p.min)*h;s+=box(cx,v,fy,colors[m],34,m)+txt(cx,y+h+19,m[0].toUpperCase()+m.slice(1),'small');});});s+=txt(485,462,'Box: IQR and median · whiskers: observed min/max · dots: individual seeds','small');save(`checkpoint_boxes_${a}.svg`,s+'</svg>');}

// Per-seed row-normalized matrices: mean in %, SD in percentage points.
const confusionRows=[];
for(const a of archs)for(const m of modes){const runs=pairs.get(`${a}/${m}`),mats=[];for(const r of runs){const rel=path.relative(root,path.join(r.run_dir,'test_predictions.csv')).replaceAll('\\','/'),pred=readCsv(rel);if(pred.length!==85)throw Error(`Expected 85 predictions: ${rel}`);const mat=classes.map(()=>classes.map(()=>0)),support=classes.map(()=>0);for(const q of pred){const i=classes.indexOf(q.true_label),j=classes.indexOf(q.predicted_label);if(i<0||j<0)throw Error('Unknown class in predictions');mat[i][j]++;support[i]++;}mats.push(mat.map((row,i)=>row.map(n=>n/support[i]*100)));}
 let s=svgStart(880,640,`${names[a]} / ${m[0].toUpperCase()+m.slice(1)}: test confusion (%)`);s+=txt(440,72,'Cell values: mean (%) ± sample SD (pp), n = 5 seeds','sub');const x0=270,y0=140,cell=109;for(let i=0;i<4;i++){s+=txt(x0+i*cell+cell/2,126,shortClassNames[i],'label');s+=txt(x0-12,y0+i*cell+cell/2+4,shortClassNames[i],'label','end');for(let j=0;j<4;j++){const vals=mats.map(mat=>mat[i][j]),mu=mean(vals),std=sd(vals),fill=theme.heatColor(mu),xx=x0+j*cell,yy=y0+i*cell;confusionRows.push({architecture:a,mode:m,true_class:classes[i],predicted_class:classes[j],n_seeds:5,mean_pct:fmt(mu,4),sd_pp:fmt(std,4)});s+=rect(xx,yy,cell-3,cell-3,fill,'stroke="#e0e6ea"')+txt(xx+cell/2,yy+49,fmt(mu,1),'sub','middle',mu>55?'style="fill:white"':'')+txt(xx+cell/2,yy+69,`± ${fmt(std,1)}`,'small','middle',mu>55?'style="fill:white"':'');}}
 s+=txt(490,608,'Rows: true class · columns: predicted class · fixed 0–100% colour scale','small');save(`confusion_${a}_${m}.svg`,s+'</svg>');}
fs.writeFileSync(path.join(__dirname,'confusion_mean_sd.csv'),['architecture,mode,true_class,predicted_class,n_seeds,mean_pct,sd_pp',...confusionRows.map(r=>Object.values(r).join(','))].join('\n')+'\n');

// Reference calibration from the saved 10-bin diagnostics, not a refit.
{
 const bins=readCsv('outputs_modes_v3_td/confidence/bin_metrics.csv').filter(r=>r.architecture==='mobilenet_v3_large'&&r.mode==='partial'&&r.seed==='42');if(bins.length!==10)throw Error(`Missing reference calibration bins: found ${bins.length}`);let s=svgStart(800,570,'Reference calibration on test: seed 42 (%)');const x=90,y=90,w=580,h=370;s+=axes(x,y,w,h,0,100,[0,20,40,60,80,100],'Mean confidence (%)')+txt(28,275,'Observed accuracy (%)','label','middle','transform="rotate(-90 28 275)"')+line(x,y+h,x+w,y,'grid','stroke-dasharray="7 5"');[0,20,40,60,80,100].forEach(v=>s+=txt(x+v/100*w,y+h+18,v,'small'));bins.forEach(b=>{if(Number(b.count)===0)return;const xx=x+Number(b.mean_confidence)*w,yy=y+h-Number(b.accuracy)*h;s+=`<circle cx="${xx}" cy="${yy}" r="${Math.max(5,Math.sqrt(Number(b.count))*2.6)}" fill="${colors.partial}" opacity=".65" stroke="#35434a"/>`+txt(xx,yy-13,b.count,'small');});s+=txt(400,532,'Dot area scales with bin count; label = images in bin (total test n = 85)','small');save('reference_calibration.svg',s+'</svg>');
}

async function brightness(){const values=readCsv('report/derived_brightness.csv').map(r=>({image_id:r.image_id,label:r.label,luminance:Number(r.mean_luminance_0_255)}));if(values.length!==560||new Set(values.map(r=>r.image_id)).size!==560||values.some(r=>!Number.isFinite(r.luminance)||!audit.some(a=>a.image_id===r.image_id&&a.label===r.label)))throw Error('Invalid saved brightness values');let s=svgStart(1150,500,'Original-image mean luminance by class (0–255)');s+=txt(290,74,'Distribution (560 decoded originals)','sub')+txt(865,74,'Image count by luminance bin','sub');const x=72,y=98,w=480,h=310;s+=axes(x,y,w,h,0,255,[0,50,100,150,200,250],'Class');classes.forEach((c,i)=>{const v=values.filter(z=>z.label===c).map(z=>z.luminance),cx=x+62+i*116,fy=t=>y+h-t/255*h;s+=box(cx,v,fy,classColors[i],40)+txt(cx,y+h+18,classNames[i].replace('San ','').replace('Tee Ju ','TJ '),'small');});const bx=640,bw=460,bins=Array.from({length:20},()=>[0,0,0,0]);values.forEach(v=>{const b=Math.min(19,Math.floor(v.luminance/256*20));bins[b][classes.indexOf(v.label)]++;});s+=axes(bx,y,bw,310,0,65,[0,15,30,45,60],'Mean luminance (0–255)');bins.forEach((row,b)=>row.forEach((n,i)=>{const xx=bx+b*bw/20+i*5.4,h=n/65*310;s+=rect(xx,y+310-h,5.2,h,classColors[i],'opacity=".65"');}));s+=txt(bx,426,'0','small')+txt(bx+bw/2,426,'128','small')+txt(bx+bw,426,'255','small');save('brightness.svg',s+'</svg>');}

brightness().then(()=>{fs.writeFileSync(path.join(__dirname,'latest_figure_sources.json'),JSON.stringify({generated_utc:new Date().toISOString(),protocol:'outputs_modes_v3_td',source_sha256:source,notes:'All model charts use 75 latest tuning and 75 latest final records. Brightness uses the saved derived_brightness.csv values from 560 originals (EXIF-aware decode, Rec.709); class colours are separate from mode colours.'},null,2)+'\n');console.log('Generated latest figures from 560 audited images, 75 tuning trials, 75 final runs.');}).catch(e=>{console.error(e);process.exitCode=1;});
