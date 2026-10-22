// Report figures from saved outputs, including the explicitly authorized eight matched CAMs.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {root,readCsv,hash,sharpRuntime,mean,sd}=require('./report_utils');
const T=require('./figure_theme');
const assets='report/report_assets/latest/';
const out='outputs_modes_v3_td/';
const sources={};
function source(p){sources[p]=hash(p);return path.join(root,p);}
function csv(p){source(p);return readCsv(p);}
const resnet=csv('report/resnet_gradcam_cases.csv');
const latestPath=out+'gradcam/mobilenet_v3_large_partial_matched_resnet/seed42/latest.json';
const latest=JSON.parse(fs.readFileSync(source(latestPath),'utf8'));
for(const [p,h]of Object.entries(latest.artifact_sha256))assert.equal(hash(out+p),h);
const mobile=csv(out+latest.directory+'/selected_cases.csv');
assert.deepEqual(resnet.map(r=>r.image_id),mobile.map(r=>r.image_id));assert.equal(mobile.length,8);
const cls={san_phraphrom:'พระพรหม',san_phraphum:'พระภูมิ',san_tayay:'ตายาย',tee_ju_eia:'ตี่จู้เอี๊ยะ'};
const txt=(x,y,v,size=20,extra='')=>'<text x="'+x+'" y="'+y+'" text-anchor="middle" font-size="'+size+'" '+extra+'>'+T.esc(v)+'</text>';
const figures=[];
async function grid(rows,title,name){
 const sharp=sharpRuntime(),width=1520,height=1750,items=[];
 let labels='<svg xmlns="http://www.w3.org/2000/svg" width="'+width+'" height="'+height+'"><style>text{font-family:'+T.fontFamily+';fill:'+T.ink+'}</style>';
 labels+=txt(width/2,40,title,31,'font-weight="700"')+txt(width/2,77,'Same eight images · seed 42 · each overlay explains that model’s predicted class',20);
 for(let i=0;i<8;i++){
   const r=rows[i],left=20+(i%2)*760,top=108+Math.floor(i/2)*405;
   labels+='<rect x="'+left+'" y="'+top+'" width="720" height="390" rx="8" fill="none" stroke="'+T.grid+'"/>';
   labels+=txt(left+360,top+29,r.image_id.split('/')[1],22,'font-weight="700"');
   for(let j=0;j<2;j++){
     const input=source(out+r.sample_directory+'/'+['original.png','overlay.png'][j]);
     items.push({input:await sharp(input).resize(330,285,{fit:'contain',background:'white'}).png().toBuffer(),left:left+20+j*350,top:top+65});
     labels+=txt(left+185+j*350,top+56,['Original','Overlay'][j],20);
   }
   labels+=txt(left+360,top+378,'จริง: '+cls[r.true_label]+' → ทำนาย: '+cls[r.predicted_label]+' | '+(100*Number(r.confidence)).toFixed(2)+'% '+(r.true_label===r.predicted_label?'✓':'✗'),21);
 }
 labels+='</svg>';items.push({input:Buffer.from(labels),left:0,top:0});
 const rel=assets+name;await sharp({create:{width,height,channels:3,background:'white'}}).composite(items).png().toFile(path.join(root,rel));figures.push(rel);
}
function reliability(){
 const bins=csv(out+'confidence/bin_metrics.csv').filter(r=>r.architecture==='mobilenet_v3_large'&&r.mode==='partial'&&r.seed==='42');
 assert.equal(bins.length,10);assert.equal(bins.reduce((s,b)=>s+Number(b.count),0),85);
 let s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 570"><rect width="800" height="570" fill="white"/><style>text{font-family:'+T.fontFamily+';fill:'+T.ink+'}</style>';
 const x=90,y=90,w=580,h=370,px=v=>x+v*w,py=v=>y+h-v*h;
 s+=txt(400,34,'MobileNetV3-Large / Partial: seed 42 (%)',22,'font-weight="700"');
 for(const t of [0,.2,.4,.6,.8,1])s+='<path d="M '+x+' '+py(t)+' H '+(x+w)+'" stroke="'+T.grid+'"/>'+txt(x-24,py(t)+5,t*100,12)+txt(px(t),y+h+20,t*100,12);
 s+='<path d="M '+x+' '+y+' V '+(y+h)+' H '+(x+w)+'" fill="none" stroke="'+T.muted+'"/><path d="M '+x+' '+(y+h)+' L '+(x+w)+' '+y+'" stroke="#c4cdd3" stroke-dasharray="7 5"/>';
 const ece=bins.reduce((s,b)=>s+Number(b.count)/85*(Number(b.count)?Math.abs(Number(b.mean_confidence)-Number(b.accuracy)):0),0)*100;
 s+=txt(265,114,'ECE (10 bins, this seed): '+ece.toFixed(2),12);
 for(const b of bins)if(Number(b.count)){const xx=px(Number(b.mean_confidence)),yy=py(Number(b.accuracy));s+='<circle cx="'+xx+'" cy="'+yy+'" r="'+Math.max(5,Math.sqrt(Number(b.count))*2.6)+'" fill="'+T.modeColors.partial+'" opacity=".65" stroke="'+T.ink+'"/>'+txt(xx,yy-13,b.count,12);}
 s+=txt(380,499,'Mean confidence (%)',14)+txt(28,275,'Observed accuracy (%)',14,'transform="rotate(-90 28 275)"')+txt(400,532,'Dot area scales with bin count; label = images in bin (test n = 85)',12)+'</svg>';
 const rel=assets+'reliability_mobilenet_v3_large_partial_seed42.svg';fs.writeFileSync(path.join(root,rel),s);figures.push(rel);return {ece_seed42_pct:ece,bins};
}
async function main(){
 const sharp=sharpRuntime();
 for(let i=0;i<8;i++){const a=await sharp(source(out+resnet[i].sample_directory+'/original.png')).raw().toBuffer({resolveWithObject:true}),b=await sharp(source(out+mobile[i].sample_directory+'/original.png')).raw().toBuffer({resolveWithObject:true});assert.deepEqual(a.info,b.info);assert(a.data.equals(b.data),'CAM originals differ');}
 await grid(resnet,'ResNet50 / Full: original and Grad-CAM overlay','gradcam_resnet50_full_original_overlay.png');
 await grid(mobile,'MobileNetV3-Large / Partial: original and Grad-CAM overlay','gradcam_mobilenet_partial_matched_original_overlay.png');
 const calibration=reliability();
 const summary=csv(out+'final/pair_summary.csv');
 const leaders=Object.fromEntries(['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'].map(k=>{const sorted=[...summary].sort((a,b)=>Number(b[k+'_mean'])-Number(a[k+'_mean']));assert.equal(sorted[0].architecture+'::'+sorted[0].mode,'resnet50::full');return [k,{leader:'resnet50::full',mean:Number(sorted[0][k+'_mean']),runner_up:sorted[1].architecture+'::'+sorted[1].mode,lead_pp:100*(Number(sorted[0][k+'_mean'])-Number(sorted[1][k+'_mean']))}];}));
 const finals=csv(out+'final/run_metrics.csv');
 const matched=[42,123,456,789,1024].map(seed=>{const find=(a,m)=>Number(finals.find(r=>r.architecture===a&&r.mode===m&&Number(r.seed)===seed).test_macro_f1)*100;return {seed,mobile_f1_pct:find('mobilenet_v3_large','partial'),resnet_f1_pct:find('resnet50','full'),difference_pp:find('mobilenet_v3_large','partial')-find('resnet50','full')};});
 const d=matched.map(r=>r.difference_pp),t=mean(d)/(sd(d)/Math.sqrt(5));
 const saved=csv('report/verified_statistics.csv').find(r=>r.reference==='mobilenet_v3_large::partial'&&r.competitor==='resnet50::full'&&r.metric==='test_macro_f1');assert(Math.abs(t-Number(saved.t))<1e-10);
 const comparison=resnet.map((r,i)=>({image_id:r.image_id,true_label:r.true_label,resnet_pred:r.predicted_label,resnet_confidence:Number(r.confidence),mobile_pred:mobile[i].predicted_label,mobile_confidence:Number(mobile[i].confidence)}));
 const keys=Object.keys(comparison[0]);fs.writeFileSync(path.join(__dirname,'matched_cam_comparison.csv'),[keys.join(','),...comparison.map(r=>keys.map(k=>r[k]).join(','))].join('\n')+'\n');
 const evidence={prepared_utc:new Date().toISOString(),leaders,paired_f1_example:{matched,mean_difference_pp:mean(d),sample_sd_difference_pp:sd(d),t,df:4,p_raw:Number(saved.p_raw),p_holm:Number(saved.p_holm),alpha:.05},calibration,confidence_pair_summary:csv(out+'confidence/pair_summary.csv'),cam_comparison:comparison,mean_selected_epochs:csv('report/mean_selected_epochs.csv'),cam_inference_authorized:true,new_training_runs:0,source_sha256:sources,output_sha256:Object.fromEntries(figures.map(p=>[p,hash(p)]))};
 fs.writeFileSync(path.join(__dirname,'readme_revision_evidence.json'),JSON.stringify(evidence,null,2)+'\n');
 console.log(JSON.stringify({figures:figures.length,all_four_metric_leaders:'ResNet50/Full',mobile_correct:mobile.filter(r=>r.true_label===r.predicted_label).length,resnet_correct:resnet.filter(r=>r.true_label===r.predicted_label).length,mobile_seed42_ece:calibration.ece_seed42_pct,paired_t:t}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
