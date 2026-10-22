// Enhanced report figures, strictly from saved audit / history / configurations.
// Run directly, or automatically after generate_latest_figures.js.
const fs = require('fs');
const path = require('path');
const {root, parseCsv, hash, mean, sd} = require('./report_utils');
const T = require('./figure_theme');
const A = ['resnet50','efficientnet_b0','mobilenet_v3_large','densenet121','convnext_tiny'];
const N = ['ResNet50','EfficientNet-B0','MobileNetV3-Large','DenseNet121','ConvNeXt-Tiny'];
const M = ['full','frozen','partial'];
const C = ['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia'];
const CN = ['San Phra Phrom','San Phra Phum','San Ta Yai','Tee Ju Eia'];
const dash = ['', '9 5', '2 5', '10 4 2 4'];
const destination = 'report/report_assets/latest/';
const sources = {}, figures = [], histograms = [];
function read(p) { sources[p]=hash(p); return fs.readFileSync(path.join(root,p),'utf8'); }
function csv(p) { return parseCsv(read(p)); }
function writeCsv(p,rows) {
  if(!rows.length)throw Error('No rows: '+p);
  const fields=Object.keys(rows[0]), cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  fs.writeFileSync(path.join(root,p),[fields.map(cell).join(','),...rows.map(r=>fields.map(f=>cell(r[f])).join(','))].join('\n')+'\n');
}
const f=(v,d=1)=>Number(v).toFixed(d);
const text=(x,y,v,size=16,extra='')=>{const anchor=extra.match(/text-anchor="([^"]+)"/);return '<text x="'+x+'" y="'+y+'" text-anchor="'+(anchor?anchor[1]:'middle')+'" font-size="'+size+'" '+extra.replace(/text-anchor="[^"]+"/,'')+'>'+T.esc(v)+'</text>';};
const line=(x,y,xx,yy,color=T.muted,extra='')=>'<line x1="'+x+'" y1="'+y+'" x2="'+xx+'" y2="'+yy+'" stroke="'+color+'" '+extra+'/>';
function frame(w,h,title,subtitle='') { return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+T.esc(title)+'"><rect width="'+w+'" height="'+h+'" fill="white"/><style>text{font-family:'+T.fontFamily+';fill:'+T.ink+'}</style>'+text(w/2,34,title,25,'font-weight="700"')+(subtitle?text(w/2,67,subtitle,16):''); }
function save(name,s) { const p=destination+name; fs.writeFileSync(path.join(root,p),s+'</svg>\n'); figures.push(p); }
function axes(b,xmin,xmax,ymin,ymax,xt,yt,xlabel,ylabel) {
  const x=v=>b.x+(v-xmin)/(xmax-xmin)*b.w, y=v=>b.y+b.h-(v-ymin)/(ymax-ymin)*b.h;
  let s='';for(const v of yt)s+=line(b.x,y(v),b.x+b.w,y(v),T.grid)+text(b.x-13,y(v)+5,Number.isInteger(v)?v:f(v),14,'text-anchor="end"');
  for(const v of xt)s+=text(x(v),b.y+b.h+24,Number.isInteger(v)?v:f(v),14);
  s+=line(b.x,b.y,b.x,b.y+b.h)+line(b.x,b.y+b.h,b.x+b.w,b.y+b.h);
  s+=text(b.x+b.w/2,b.y+b.h+55,xlabel,17)+text(b.x-56,b.y+b.h/2,ylabel,16,'transform="rotate(-90 '+(b.x-56)+' '+(b.y+b.h/2)+')"');
  return {s,x,y};
}
function marker(x,y,m,r=4,opacity=1) {
  const color=T.modeColors[m];
  if(m==='partial')return '<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+(2*r)+'" height="'+(2*r)+'" fill="'+color+'" opacity="'+opacity+'"/>';
  if(m==='full')return '<path d="M '+x+' '+(y-r)+' l '+r+' '+r+' l '+(-r)+' '+r+' l '+(-r)+' '+(-r)+' Z" fill="'+color+'" opacity="'+opacity+'"/>';
  return '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+color+'" opacity="'+opacity+'"/>';
}
function classLegend(y,w=1440) {let s='';C.forEach((_,i)=>{const x=135+i*(w-100)/4;s+=line(x-50,y-5,x-12,y-5,T.classColors[i],'stroke-width="4" stroke-dasharray="'+dash[i]+'"')+text(x,y,CN[i],15,'text-anchor="start"');});return s;}
function proportions(values,lo,hi,step,label) {
  const bins=Array.from({length:Math.round((hi-lo)/step)},()=>[0,0,0,0]);
  for(const r of values){if(r.value<lo||r.value>hi)throw Error('Histogram would omit observation');const i=Math.min(bins.length-1,Math.floor((r.value-lo)/step+1e-10));bins[i][C.indexOf(r.label)]++;}
  const totals=C.map(c=>values.filter(r=>r.label===c).length);
  bins.forEach((counts,i)=>counts.forEach((n,c)=>histograms.push({figure:label,class:C[c],bin_left:lo+i*step,bin_right:lo+(i+1)*step,count:n,class_total:totals[c],proportion_pct:n/totals[c]*100})));
  return {bins,totals};
}
function histogram(b,values,lo,hi,step,xlabel,label,meanLines=false) {
  const {bins,totals}=proportions(values,lo,hi,step,label);
  const max=Math.max(...bins.flatMap(row=>row.map((n,c)=>n/totals[c]*100))), ymax=Math.ceil(max/10)*10;
  const xt=[lo,...[0.5,1,1.5,2].filter(v=>v>lo&&v<hi),hi];
  const ticks=xlabel.includes('brightness')?[0,64,128,192,255]:xt;
  const p=axes(b,lo,hi,0,ymax,ticks,Array.from({length:ymax/10+1},(_,i)=>i*10),xlabel,'Proportion of images (%)');let s=p.s;
  for(let c=0;c<4;c++){
    const points=[];for(let i=0;i<bins.length;i++){const y=p.y(bins[i][c]/totals[c]*100);points.push([p.x(lo+i*step),y],[p.x(lo+(i+1)*step),y]);}
    const d='M '+p.x(lo)+' '+p.y(0)+' L '+points.map(v=>v.join(' ')).join(' L ')+' L '+p.x(hi)+' '+p.y(0)+' Z';
    s+='<path d="'+d+'" fill="'+T.classColors[c]+'" fill-opacity=".25" stroke="'+T.classColors[c]+'" stroke-width="2.5" stroke-dasharray="'+dash[c]+'"/>';
    if(meanLines){const mu=mean(values.filter(r=>r.label===C[c]).map(r=>r.value));s+=line(p.x(mu),b.y,p.x(mu),b.y+b.h,T.classColors[c],'stroke-width="2" stroke-dasharray="'+(dash[c]||'4 3')+'"');}
  }
  if(!meanLines)s+=line(p.x(1),b.y,p.x(1),b.y+b.h,T.ink,'stroke-width="2" stroke-dasharray="7 5"')+text(p.x(1)+8,b.y+20,'Square (1:1)',14,'text-anchor="start"');
  return s;
}
function datasetFigures(audit) {
  let s=frame(1100,500,'Class distribution','560 original images · fixed class order');
  const p=axes({x:100,y:100,w:940,h:300},0,4,0,220,[],[0,50,100,150,200],'Class','Number of images');s+=p.s;
  C.forEach((c,i)=>{const n=audit.filter(r=>r.label===c).length,x=p.x(i+.5);s+='<rect x="'+(x-65)+'" y="'+p.y(n)+'" width="130" height="'+(p.y(0)-p.y(n))+'" fill="'+T.classColors[i]+'"/>'+text(x,p.y(n)-12,n,19)+text(x,425,CN[i],16);});save('class_distribution.svg',s);
  s=frame(1440,650,'Image dimensions and aspect ratio','Original decoded dimensions · histogram uses common bins and within-class percentages');
  const a=axes({x:100,y:120,w:550,h:400},0,4500,0,4500,[0,1000,2000,3000,4000],[0,1000,2000,3000,4000],'Width (px)','Height (px)');s+=a.s+line(a.x(0),a.y(0),a.x(4500),a.y(4500),T.muted,'stroke-dasharray="7 5"');
  audit.forEach(r=>{const c=C.indexOf(r.label),xx=a.x(Number(r.width)),yy=a.y(Number(r.height));s+=['<circle cx="'+xx+'" cy="'+yy+'" r="3"/>','<rect x="'+(xx-3)+'" y="'+(yy-3)+'" width="6" height="6"/>','<path d="M '+xx+' '+(yy-4)+' l 4 7 h -8 Z"/>','<path d="M '+(xx-3)+' '+(yy-3)+' l 6 6 M '+(xx+3)+' '+(yy-3)+' l -6 6"/>'][c].replace('/>',' fill="'+T.classColors[c]+'" stroke="'+T.classColors[c]+'" opacity=".5"/>');});
  const vals=audit.map(r=>({label:r.label,value:Number(r.width)/Number(r.height)}));const lo=Math.floor(Math.min(...vals.map(r=>r.value))*10)/10,hi=Math.ceil(Math.max(...vals.map(r=>r.value))*10)/10;
  s+=histogram({x:810,y:120,w:570,h:400},vals,lo,hi,.1,'Aspect ratio (width / height)','aspect_ratio')+classLegend(620);save('image_dimensions.svg',s);
  s=frame(1330,620,'Decoded colour modes','Direct counts and percentages; non-RGB modes shown separately at an expanded scale');
  const modes=['RGB','RGBA','P','CMYK'], labels=['RGB','RGBA','Palette','CMYK'],patterns=['','////','....','xxxx'];
  s+='<defs>'+modes.map((m,i)=>'<pattern id="pattern'+i+'" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="'+['#374151','#D1D5DB','#F3F4F6','#9CA3AF'][i]+'"/>'+(['','<path d="M 0 8 L 8 0" stroke="#374151"/>','<circle cx="4" cy="4" r="1.5" fill="#374151"/>','<path d="M 0 0 L 8 8 M 8 0 L 0 8" stroke="#374151"/>'][i])+'</pattern>').join('')+'</defs>';
  for(let panel=0;panel<2;panel++){const b={x:130+panel*650,y:120,w:440,h:340},max=panel?10:560,series=panel?[1,2,3]:[0,1,2,3];s+=text(b.x+b.w/2,100,panel?'Non-RGB detail':'All images',19,'font-weight="700"');
    s+=line(b.x,b.y,b.x,b.y+b.h)+line(b.x,b.y+b.h,b.x+b.w,b.y+b.h)+text(b.x+b.w/2,520,'Number of images',17)+text(b.x-100,b.y+b.h/2,'Colour mode',17,'transform="rotate(-90 '+(b.x-100)+' '+(b.y+b.h/2)+')"');
    for(const t of (panel?[0,2,4,6,8,10]:[0,100,200,300,400,500]))s+=line(b.x+t/max*b.w,b.y,b.x+t/max*b.w,b.y+b.h,T.grid)+text(b.x+t/max*b.w,485,t,14);
    series.forEach((idx,k)=>{const count=audit.filter(r=>r.mode===modes[idx]).length,yy=b.y+30+k*78,ww=count/max*b.w;s+='<rect x="'+b.x+'" y="'+yy+'" width="'+ww+'" height="42" fill="url(#pattern'+idx+')" stroke="#374151"/>'+text(b.x-12,yy+27,labels[idx],16,'text-anchor="end"')+text(b.x+ww+10,yy+27,count+' ('+f(count/560*100,2)+')',16,'text-anchor="start"');});}
  s+=text(665,579,'Labels: image count (percentage of all 560 images); patterns distinguish colour modes without hue',16);save('colour_modes.svg',s);
  const v=csv('report/derived_brightness.csv').map(r=>({label:r.label,value:Number(r.mean_luminance_0_255)}));
  if(v.length!==560)throw Error('Brightness evidence incomplete');
  s=frame(1440,680,'Image brightness by class','One value per original image · approximate RGB-weighted brightness (0–255)');
  const bp=axes({x:110,y:120,w:540,h:410},0,4,0,255,[],[0,50,100,150,200,250],'Class','Mean image brightness (0–255)');s+=bp.s;
  const q=(a,p)=>{const b=[...a].sort((x,y)=>x-y),k=(b.length-1)*p,j=Math.floor(k);return b[j]+(b[Math.ceil(k)]-b[j])*(k-j);};
  C.forEach((c,i)=>{const vs=v.filter(r=>r.label===c).map(r=>r.value),cx=bp.x(i+.5),lo=Math.min(...vs),hi=Math.max(...vs),q1=q(vs,.25),q2=q(vs,.5),q3=q(vs,.75),mu=mean(vs);s+=line(cx,bp.y(lo),cx,bp.y(hi),T.classColors[i],'stroke-width="2"')+line(cx-12,bp.y(lo),cx+12,bp.y(lo))+line(cx-12,bp.y(hi),cx+12,bp.y(hi))+'<rect x="'+(cx-38)+'" y="'+bp.y(q3)+'" width="76" height="'+(bp.y(q1)-bp.y(q3))+'" fill="'+T.classColors[i]+'" fill-opacity=".35" stroke="'+T.ink+'"/>'+line(cx-38,bp.y(q2),cx+38,bp.y(q2),T.ink,'stroke-width="3"')+line(cx-7,bp.y(mu)-7,cx+7,bp.y(mu)+7,T.ink,'stroke-width="3"')+line(cx-7,bp.y(mu)+7,cx+7,bp.y(mu)-7,T.ink,'stroke-width="3"')+text(cx,556,CN[i].replace('San ',''),15);});
  const globalMean=mean(v.map(r=>r.value));s+=line(110,bp.y(globalMean),650,bp.y(globalMean),T.muted,'stroke-dasharray="8 5" stroke-width="2"');
  s+=histogram({x:810,y:120,w:570,h:410},v,0,255,12.75,'Mean image brightness (0–255)','brightness',true)+classLegend(623);
  s+=text(720,660,'Box: IQR · thick line: median · ×: class mean · horizontal dashed line: overall mean · histogram vertical lines: class means',15);save('brightness.svg',s);
  writeCsv('report/image_property_histograms.csv',histograms);
  const colorRows=C.flatMap(c=>modes.map(m=>({class:c,colour_mode:m,count:audit.filter(r=>r.label===c&&r.mode===m).length})));writeCsv('report/colour_mode_counts.csv',colorRows);
}
function loadHistories(registry) {
  const records=[];
  for(const stage of ['trials','final_runs'])for(const r of registry[stage].filter(r=>r.status==='completed')){
    const directory=path.relative(root,r.run_dir).replaceAll('\\','/');
    if(!directory.startsWith('outputs_modes_v3_td/'))throw Error('Historical result mixing');
    const historyFile=directory+'/history.csv';const rows=csv(historyFile).filter(h=>h.train_loss!=='').map(h=>Object.fromEntries(Object.entries(h).filter(([k])=>['epoch','train_loss','val_loss','train_acc','val_acc','val_macro_f1'].includes(k)).map(([k,v])=>[k,Number(v)])));
    if(rows.some((h,i)=>h.epoch!==i+1||Object.values(h).some(v=>!Number.isFinite(v))))throw Error('Invalid history '+historyFile);
    const selected=rows.find(h=>h.epoch===Number(r.best_epoch));if(!selected||Math.abs(selected.val_macro_f1-Number(r.best_val_macro_f1))>1e-10)throw Error('Selected epoch mismatch');
    records.push({...r,stage,rows,selected,historyFile,history_sha256:sources[historyFile]});
  }
  if(records.filter(r=>r.stage==='trials').length!==75||records.filter(r=>r.stage==='final_runs').length!==75)throw Error('Expected 75+75 completed records');
  return records;
}
function tuningFigures(records,winners) {
  const trials=records.filter(r=>r.stage==='trials');
  const metrics=[['learning_rate','Learning rate (log scale)',true],['weight_decay','Weight decay (log scale)',true],['train_loss','Training loss at selected epoch',false],['val_loss','Validation loss at selected epoch',false],['epoch','Selected epoch',false],['train_acc','Training accuracy (%)',false],['val_acc','Validation accuracy (%)',false]];
  const rows=trials.map(r=>({architecture:r.architecture,mode:r.mode,trial:r.trial,seed:r.seed,identity:r.identity,learning_rate:r.config.learning_rate,weight_decay:r.config.weight_decay,...r.selected,winner:Number(winners[r.architecture+'::'+r.mode].winning_trial)===Number(r.trial),history_file:r.historyFile,history_sha256:r.history_sha256}));
  writeCsv('report/tuning_checkpoint_observations.csv',rows);
  let s=frame(1440,1540,'Hyperparameter search and selected-epoch diagnostics','75 trials · tuning seed 42 · rings mark 15 within-setup winners · no test scores used');
  metrics.forEach(([key,label,log],i)=>{const b={x:95+(i%2)*720,y:135+Math.floor(i/2)*345,w:570,h:225};
    const vals=rows.map(r=>Number(r[key])*(key.endsWith('_acc')?100:1)),min=log?(key==='learning_rate'?1e-5:1e-6):0,max=log?(key==='learning_rate'?1e-3:1e-2):key.endsWith('_acc')?100:key==='epoch'?30:Math.ceil(Math.max(...vals)*10)/10;
    const xl=log?Math.log10(min):min,xh=log?Math.log10(max):max,xt=log?[]:[0,max/2,max],p=axes(b,xl,xh,40,100,xt,[40,60,80,100],label,'Validation Macro F1 (%)');s+=p.s;
    if(log)for(let v=Math.log10(min);v<=Math.log10(max);v++)s+=text(p.x(v),b.y+b.h+24,'10^'+v,14);
    rows.forEach((r,j)=>{const xx=p.x(log?Math.log10(vals[j]):vals[j]),yy=p.y(r.val_macro_f1*100);s+=marker(xx,yy,r.mode,4,.65);if(r.winner)s+='<circle cx="'+xx+'" cy="'+yy+'" r="7" fill="none" stroke="'+T.ink+'" stroke-width="1.5"/>';});
  });
  let lx=850;for(const m of ['frozen','partial','full']){s+=marker(lx,1290,m,7)+text(lx+18,1296,m[0].toUpperCase()+m.slice(1),18,'text-anchor="start"');lx+=170;}
  s+=text(1080,1350,'Loss / accuracy values share the selected epoch',17)+text(1080,1382,'These are associations, not causal effects',17)+text(1080,1414,'15 setups = 5 architectures × 3 strategies',17);save('tuning_response.svg',s);
  s=frame(1440,610,'Tuning trials within each predeclared setup','5 configurations per setup · ring = selected recipe · points are configurations, not repeated seeds');
  const p=axes({x:90,y:120,w:1290,h:310},0,15,40,100,[],[40,50,60,70,80,90,100],'','Validation Macro F1 (%)');s+=p.s;
  A.forEach((a,ai)=>M.forEach((m,mi)=>{const x=ai*3+mi+.5;rows.filter(r=>r.architecture===a&&r.mode===m).forEach((r,j)=>{const xx=p.x(x+(j-2)*.045),yy=p.y(r.val_macro_f1*100);s+=marker(xx,yy,m,5,.6);if(r.winner)s+='<circle cx="'+xx+'" cy="'+yy+'" r="9" fill="none" stroke="'+T.ink+'" stroke-width="2"/>';});s+=text(p.x(x),454,m,14);if(mi===1)s+=text(p.x(x),485,N[ai].replace('MobileNetV3-','MobV3-').replace('EfficientNet-','EffNet-'),16);}));
  s+=text(720,535,'Architecture × fine-tuning strategy',17)+text(720,580,'Each setup selects its own recipe using validation Macro F1; none is screened out by Full-only results',16);save('tuning_setup_summary.svg',s);
}
function learningFigures(records) {
  const finals=records.filter(r=>r.stage==='final_runs'), aggregates=[],checkpoints=[];
  const keys=['train_loss','val_loss','train_acc','val_acc','val_macro_f1'];
  for(const a of A)for(const m of M){const runs=finals.filter(r=>r.architecture===a&&r.mode===m).sort((x,y)=>Number(x.seed)-Number(y.seed));
    if(runs.length!==5||runs.map(r=>Number(r.seed)).join(',')!=='42,123,456,789,1024')throw Error('Missing matched seeds');
    for(const r of runs)checkpoints.push({architecture:a,mode:m,seed:r.seed,run_id:r.identity,selected_epoch:r.best_epoch,stopped_epoch:r.rows.length,validation_macro_f1:r.selected.val_macro_f1,validation_accuracy:r.selected.val_acc,checkpoint:path.relative(root,r.checkpoint).replaceAll('\\','/'),history_file:r.historyFile,history_sha256:r.history_sha256});
    for(let e=1;e<=Math.max(...runs.map(r=>r.rows.length));e++){const available=runs.filter(r=>r.rows.some(h=>h.epoch===e));for(const key of keys){const values=available.map(r=>r.rows.find(h=>h.epoch===e)[key]);aggregates.push({architecture:a,mode:m,epoch:e,metric:key,n:values.length,seeds:available.map(r=>r.seed).join(';'),mean:mean(values),sample_sd:values.length>1?sd(values):'',unit:key.includes('loss')?'weighted_cross_entropy':'fraction'});}}
  }
  writeCsv('report/learning_curve_summary.csv',aggregates);writeCsv('report/final_checkpoint_summary.csv',checkpoints);
  writeCsv('report/mean_selected_epochs.csv', A.flatMap(a=>M.map(m=>({architecture:a,mode:m,n:5,mean_selected_epoch:mean(checkpoints.filter(r=>r.architecture===a&&r.mode===m).map(r=>Number(r.selected_epoch))),seed_ids:'42;123;456;789;1024'}))));
  const losses=aggregates.filter(r=>r.metric.includes('loss'));const lossMin=Math.min(0,...losses.filter(r=>r.sample_sd!=='').map(r=>r.mean-r.sample_sd));const lossMax=Math.max(...losses.map(r=>r.mean+(r.sample_sd===''?0:r.sample_sd)));
  const lossLo=Math.floor(lossMin*2)/2,lossHi=Math.ceil(lossMax*2)/2;
  const fractions=aggregates.filter(r=>!r.metric.includes('loss'));
  const pctLo=Math.floor(Math.min(0,...fractions.map(r=>(r.mean-(r.sample_sd===''?0:r.sample_sd))*100))/5)*5;
  const pctHi=Math.ceil(Math.max(100,...fractions.map(r=>(r.mean+(r.sample_sd===''?0:r.sample_sd))*100))/5)*5;
  A.forEach((a,ai)=>{
    let s=frame(1560,1450,N[ai]+': learning curves across five matched seeds','Solid: training · dashed: validation · shading: ±1 sample SD · ×: each run’s selected validation checkpoint');
    M.forEach((m,mi)=>{s+=marker(200+mi*510,99,m,7)+text(225+mi*510,105,m[0].toUpperCase()+m.slice(1),19,'text-anchor="start"');
      const rows=[{keys:['train_loss','val_loss'],label:'Weighted cross-entropy loss',lo:lossLo,hi:lossHi,scale:1},{keys:['train_acc','val_acc'],label:'Accuracy (%)',lo:pctLo,hi:pctHi,scale:100},{keys:['val_macro_f1'],label:'Validation Macro F1 (%)',lo:pctLo,hi:pctHi,scale:100},{keys:[],label:'Available runs',lo:0,hi:5,scale:1}];
      rows.forEach((spec,ri)=>{const b={x:90+mi*510,y:140+ri*315,w:400,h:205},p=axes(b,1,30,spec.lo,spec.hi,[1,5,10,15,20,25,30],ri===0?[spec.lo,(spec.lo+spec.hi)/2,spec.hi]:ri===3?[0,1,2,3,4,5]:[0,25,50,75,100],'Epoch',spec.label);s+=p.s;
        const get=key=>aggregates.filter(r=>r.architecture===a&&r.mode===m&&r.metric===key);
        if(ri===3){const data=get('val_macro_f1');let d='M '+p.x(1)+' '+p.y(data[0].n);for(let i=1;i<data.length;i++)d+=' H '+p.x(data[i].epoch)+' V '+p.y(data[i].n);const last=data[data.length-1].epoch;if(last<30)d+=' H '+p.x(last+1)+' V '+p.y(0)+' H '+p.x(30);s+='<path d="'+d+'" fill="none" stroke="'+T.modeColors[m]+'" stroke-width="3"/>';return;}
        spec.keys.forEach((key,k)=>{const data=get(key),band=data.filter(r=>r.n>=2);if(band.length){const upper=band.map(r=>[p.x(r.epoch),p.y((r.mean+r.sample_sd)*spec.scale)]),lower=band.map(r=>[p.x(r.epoch),p.y((r.mean-r.sample_sd)*spec.scale)]).reverse();s+='<path d="M '+[...upper,...lower].map(v=>v.join(' ')).join(' L ')+' Z" fill="'+T.modeColors[m]+'" opacity="'+(k===0?.13:.23)+'"/>';}
          s+='<path d="M '+data.map(r=>p.x(r.epoch)+' '+p.y(r.mean*spec.scale)).join(' L ')+'" fill="none" stroke="'+T.modeColors[m]+'" stroke-width="3" stroke-dasharray="'+(key.startsWith('val')?'9 5':'')+'"/>';
        });
        if(ri===2)finals.filter(r=>r.architecture===a&&r.mode===m).forEach(r=>{const xx=p.x(r.best_epoch),yy=p.y(r.selected.val_macro_f1*100);s+=line(xx-4,yy-4,xx+4,yy+4,T.ink,'stroke-width="2"')+line(xx-4,yy+4,xx+4,yy-4,T.ink,'stroke-width="2"');});
        if(ri===2){const selectedMean=mean(finals.filter(r=>r.architecture===a&&r.mode===m).map(r=>Number(r.best_epoch)));s+=line(p.x(selectedMean),b.y,p.x(selectedMean),b.y+b.h,T.ink,'stroke-width="2" stroke-dasharray="3 5"')+text(b.x+b.w-8,b.y+b.h-16,'Mean selected epoch = '+selectedMean.toFixed(1),15,'text-anchor="end" font-weight="600"');}
      });
    });
    s+=text(780,1424,'Observed epochs only; no zero-fill or carry-forward · n = 1: mean only, no sample SD · common axes across architectures',16);save('learning_curves_'+a+'.svg',s);
  });
}
async function generate() {
  read('report/figure_theme.js');const audit=csv('outputs_modes_v3_td/dataset_audit/image_properties.csv');if(audit.length!==560)throw Error('Expected 560 audited originals');
  const registry=JSON.parse(read('outputs_modes_v3_td/run_registry.json'));const winners=JSON.parse(read('outputs_modes_v3_td/tuning/best_params_by_arch_mode.json')).pairs;
  const records=loadHistories(registry);datasetFigures(audit);tuningFigures(records,winners);learningFigures(records);
  for(const [p,h]of Object.entries(sources))if(hash(p)!==h)throw Error('Source changed during rendering: '+p);
  fs.writeFileSync(path.join(__dirname,'enhanced_figure_sources.json'),JSON.stringify({generated_utc:new Date().toISOString(),experiment:'outputs_modes_v3_td',source_sha256:sources,output_sha256:Object.fromEntries(figures.map(p=>[p,hash(p)])),tuning_trials:75,final_runs:75,final_pairs:15,sample_sd_ddof:1,histogram_unit:'within-class percentage per bin; not probability density',learning_curves:'Observed epochs only; sample SD omitted at n=1; train metrics logged during augmented training, validation after each epoch.',python_inference_training_performed:false},null,2)+'\n');
  console.log('Generated enhanced EDA, seven tuning diagnostics and learning curves for all 15 five-seed setups.');
}
module.exports={generate};
if(require.main===module)generate().catch(e=>{console.error(e);process.exitCode=1;});
