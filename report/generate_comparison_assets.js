// Rebuild approved README CAM layout and a separate two-model loss figure from saved evidence.
// Run: node report/generate_comparison_assets.js. No Python/model execution.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..');
const {readCsv,hash,sharpRuntime,mean,sd}=require(path.join(root,'report/report_utils'));
const T=require(path.join(root,'report/figure_theme'));
const sharp=sharpRuntime(), assets='report/report_assets/latest/', standalone='report/report_assets/standalone/';
for(const p of [assets,standalone])fs.mkdirSync(path.join(root,p),{recursive:true});
const protectedPaths=['outputs_modes_v3_td/selection.json','outputs_modes_v3_td/run_registry.json'];
const before=Object.fromEntries(protectedPaths.map(p=>[p,hash(p)]));
const inputs={},outputs=[],source=p=>{inputs[p]=hash(p);return path.join(root,p);};
function csv(p){source(p);return readCsv(p);}
function json(p){return JSON.parse(fs.readFileSync(source(p),'utf8'));}
const txt=(x,y,v,size=18,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" ${extra.includes('text-anchor=')?'':'text-anchor="middle"'} ${extra}>${T.esc(v)}</text>`;
const line=(x,y,x2,y2,color=T.muted,extra='')=>`<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
const frame=(w,h,title,sub)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="${w}" height="${h}" fill="white"/><style>text{font-family:${T.fontFamily};fill:${T.ink}}</style>${txt(w/2,35,title,26,'font-weight="700"')}${txt(w/2,65,sub,16)}`;
async function saveSvg(rel,svg){fs.writeFileSync(path.join(root,rel),svg);outputs.push(rel);const png=rel.replace(/\.svg$/,'.png');await sharp(Buffer.from(svg)).png().toFile(path.join(root,png));outputs.push(png);}
const summaries=csv('report/learning_curve_summary.csv');
const checkpoints=csv('report/final_checkpoint_summary.csv');
const selected=csv('report/mean_selected_epochs.csv');
const setups=[['resnet50','full','ResNet50 / Full'],['mobilenet_v3_large','partial','MobileNetV3-Large / Partial']];
async function lossFigure(){
  const get=(a,m,key)=>summaries.filter(r=>r.architecture===a&&r.mode===m&&r.metric===key).map(r=>({...r,epoch:+r.epoch,n:+r.n,mean:+r.mean,sample_sd:r.sample_sd===''?null:+r.sample_sd}));
  const all=setups.flatMap(([a,m])=>['train_loss','val_loss'].flatMap(k=>get(a,m,k)));
  const hi=Math.ceil(Math.max(...all.map(r=>r.mean+(r.sample_sd??0)))*5)/5,lo=Math.min(0,Math.floor(Math.min(...all.map(r=>r.mean-(r.sample_sd??0)))*5)/5);
  let s=frame(1440,700,'Training and validation loss across five seeds','Solid: training mean · dashed: validation mean · shading: ±1 sample SD (ddof = 1)');
  const values=[];
  for(const [i,[a,m,title]] of setups.entries()){
    const b={x:100+i*700,y:132,w:570,h:310},x=e=>b.x+(e-1)/29*b.w,y=v=>b.y+b.h-(v-lo)/(hi-lo)*b.h;
    const runs=checkpoints.filter(r=>r.architecture===a&&r.mode===m);assert.equal(runs.length,5);
    for(const r of runs){assert.equal(hash(r.history_file),r.history_sha256);source(r.history_file);}
    const mu=mean(runs.map(r=>+r.selected_epoch));assert.equal(mu,+selected.find(r=>r.architecture===a&&r.mode===m).mean_selected_epoch);
    s+=txt(b.x+b.w/2,105,title,23,'font-weight="700"');
    for(let j=0;j<=5;j++){const v=lo+(hi-lo)*j/5;s+=line(b.x,y(v),b.x+b.w,y(v),T.grid)+txt(b.x-17,y(v)+5,v.toFixed(2),14,'text-anchor="end"');}
    for(const e of [1,5,10,15,20,25,30])s+=txt(x(e),b.y+b.h+25,e,14);
    s+=line(b.x,b.y,b.x,b.y+b.h)+line(b.x,b.y+b.h,b.x+b.w,b.y+b.h)+txt(b.x-65,b.y+b.h/2,'Weighted cross-entropy loss',17,`transform="rotate(-90 ${b.x-65} ${b.y+b.h/2})"`);
    for(const [k,key] of ['train_loss','val_loss'].entries()){
      const data=get(a,m,key),band=data.filter(r=>r.n>=2);values.push(...data.map(r=>({architecture:a,mode:m,...r})));
      const historyRows=runs.map(r=>csv(r.history_file).filter(h=>h.train_loss!==''));
      for(const r of data){const vs=historyRows.map(rows=>rows.find(h=>+h.epoch===r.epoch)).filter(Boolean).map(h=>+h[key]);assert.equal(vs.length,r.n);assert(Math.abs(mean(vs)-r.mean)<1e-12);if(vs.length>1)assert(Math.abs(sd(vs)-r.sample_sd)<1e-12);}
      if(band.length){const points=[...band.map(r=>[x(r.epoch),y(r.mean+r.sample_sd)]),...band.map(r=>[x(r.epoch),y(r.mean-r.sample_sd)]).reverse()];s+=`<path d="M ${points.map(p=>p.join(' ')).join(' L ')} Z" fill="${T.modeColors[m]}" opacity="${k===0?.13:.23}"/>`;}
      s+=`<path d="M ${data.map(r=>x(r.epoch)+' '+y(r.mean)).join(' L ')}" fill="none" stroke="${T.modeColors[m]}" stroke-width="3" stroke-dasharray="${k?'9 5':''}"/>`;
    }
    s+=line(x(mu),b.y,x(mu),b.y+b.h,T.ink,'stroke-width="2" stroke-dasharray="3 5"')+txt(b.x+b.w-8,b.y+b.h-15,'Mean selected epoch = '+mu.toFixed(1),16,'text-anchor="end" font-weight="600"');
    const ns=get(a,m,'val_loss'),ny=n=>560-n*10;
    s+=txt(b.x-15,ny(2.5),'n',15,'text-anchor="end"');
    for(const n of [0,5])s+=txt(b.x-15,ny(n)+5,n,13,'text-anchor="end"')+line(b.x,ny(n),b.x+b.w,ny(n),T.grid);
    let d='M '+x(1)+' '+ny(ns[0].n);for(let j=1;j<ns.length;j++)d+=' H '+x(ns[j].epoch)+' V '+ny(ns[j].n);if(ns.at(-1).epoch<30)d+=' H '+x(ns.at(-1).epoch+1)+' V '+ny(0)+' H '+x(30);
    s+=`<path d="${d}" fill="none" stroke="${T.modeColors[m]}" stroke-width="2"/>`;
    s+=txt(b.x+b.w/2,594,'Epoch · lower trace = available runs',16);
  }
  s+=txt(720,631,'Only observed epochs are averaged; no zero-fill or carry-forward. n = 1: mean only, no SD.',16)+txt(720,659,'Mean selected epoch averages five validation-selected checkpoints; it is not a shared stopping epoch.',16)+txt(720,684,'Training includes augmentation; validation uses deterministic preprocessing. Seeds: 42, 123, 456, 789, 1024.',15)+'</svg>';
  await saveSvg(standalone+'resnet_full_mobilenet_partial_loss_five_seeds.svg',s);
}
const out='outputs_modes_v3_td/';
const res=csv('report/resnet_gradcam_cases.csv');
const matched=json(out+'gradcam/mobilenet_v3_large_partial_matched_resnet/seed42/latest.json');
const mob=csv(out+matched.directory+'/selected_cases.csv');
const wrongLatest=json(out+'gradcam/misclassified_overlays_v1/seed42/latest.json');
const wrong=csv(out+wrongLatest.directory+'/all_misclassified_cases.csv');
const counterparts=json(out+'gradcam/matched_error_counterparts_v1/seed42/latest.json');
const fullErrorCases=csv(out+counterparts.directory+'/matched_error_cases.csv');
assert.equal(fullErrorCases.length,14);assert.equal(counterparts.new_inference_cases,4);
for(const [p,h]of Object.entries(counterparts.artifact_sha256))if(!p.endsWith('.npy'))assert.equal(hash(out+p),h);
const classNames={san_phraphrom:'Phra Phrom',san_phraphum:'Phra Phum',san_tayay:'Ta Yai',tee_ju_eia:'Tee Ju Eia'};
const correctIds=['san_phraphrom/san_phraphrom_0051','san_phraphum/san_phraphum_0065','san_tayay/san_tayay_0078','tee_ju_eia/tee_ju_eia_0050'];
const correctRows=correctIds.map(id=>({id,res:res.find(r=>r.image_id===id),mob:mob.find(r=>r.image_id===id)}));
for(const r of correctRows){assert.equal(r.res.true_label,r.res.predicted_label);assert.equal(r.mob.true_label,r.mob.predicted_label);}
const wrongIds=[...new Set(wrong.map(r=>r.image_id))].sort();assert.equal(wrongIds.length,7);assert.equal(wrong.length,9);
const wrongRows=wrongIds.map(id=>({id,res:fullErrorCases.find(r=>r.image_id===id&&r.architecture==='resnet50'),mob:fullErrorCases.find(r=>r.image_id===id&&r.architecture==='mobilenet_v3_large')}));
for(const r of wrongRows)assert(r.res&&r.mob,'Both model overlays are required for '+r.id);
const placements=[];
async function grid(rows,title,name,isWrong=false,compact=false){
  const width=1360,rowH=compact?308:374,imgH=compact?225:285,height=135+rowH*rows.length+50,items=[];
  let labels=frame(width,height,title,'Test partition · seed 42 · each overlay explains that model’s saved predicted class').replace('<rect width="1360" height="'+height+'" fill="white"/>','');
  ['Original','ResNet50 / Full','MobileNetV3-Large / Partial'].forEach((v,i)=>labels+=txt(240+i*440,105,v,23,'font-weight="700"'));
  for(const [i,r] of rows.entries()){
    const reference=r.res||r.mob,top=128+i*rowH;
    labels+=`<line x1="20" y1="${top-8}" x2="1340" y2="${top-8}" stroke="${T.grid}"/>`;
    for(const [j,caseRow] of [reference,r.res,r.mob].entries()){
      const left=30+j*440,center=left+210;
      if(caseRow){
        const rel=out+caseRow.sample_directory+'/'+(j===0?'original.png':'overlay.png');
        items.push({input:await sharp(source(rel)).resize(400,imgH,{fit:'contain',background:'white'}).png().toBuffer(),left,top});
        if(j>0)placements.push({figure:assets+name,image_id:r.id,model:j===1?'resnet50/full':'mobilenet_v3_large/partial',source:rel,sha256:hash(rel)});
      }
      if(j===0){labels+=txt(center,top+imgH+23,r.id.split('/')[1],18,'font-weight="600"')+txt(center,top+imgH+48,'True: '+classNames[reference.true_label],18);}
      else if(caseRow){const incorrect=caseRow.predicted_label!==caseRow.true_label;labels+=txt(center,top+imgH+23,'Predicted: '+classNames[caseRow.predicted_label],19,'font-weight="600"')+txt(center,top+imgH+48,(+caseRow.confidence*100).toFixed(2)+'% · '+(incorrect?'WRONG':'CORRECT'),18,`style="fill:${incorrect?'#B42318':'#15803D'}"`);}
      else {throw Error('Missing matched overlay: '+r.id);}
    }
    if(r.res&&r.mob){const p=out+r.res.sample_directory+'/original.png',q=out+r.mob.sample_directory+'/original.png';const a=await sharp(source(p)).raw().toBuffer({resolveWithObject:true}),b=await sharp(source(q)).raw().toBuffer({resolveWithObject:true});assert.deepEqual(a.info,b.info);assert(a.data.equals(b.data));}
  }
  labels+=txt(width/2,height-18,isWrong?'Seven images misclassified by either model; both overlays shown. Green: correct · red: wrong.':(compact?'One correct example shown; the complete figure contains one existing matched example per class.':'One existing matched, jointly correct example per class; these examples do not estimate overall performance.'),16)+'</svg>';
  items.push({input:Buffer.from(labels),left:0,top:0});const rel=assets+name;
  await sharp({create:{width,height,channels:3,background:'white'}}).composite(items).png().toFile(path.join(root,rel));outputs.push(rel);
}

async function main(){
  const registry=json(out+'run_registry.json'), predictions={};
  for(const [a,m]of setups){
    const runs=registry.final_runs.filter(r=>r.architecture===a&&r.mode===m&&r.seed===42&&r.status==='completed');assert.equal(runs.length,1);
    const rel=path.relative(root,runs[0].run_dir).replaceAll('\\','/')+'/test_predictions.csv';assert(rel.startsWith(out+'final/'));
    const rows=csv(rel);assert.equal(rows.length,85);assert.equal(new Set(rows.map(r=>r.image_id)).size,85);
    const key=a+'_'+m;assert.equal(hash(rel),wrongLatest.inputs[key].prediction_sha256);predictions[a]=new Map(rows.map(r=>[r.image_id,r]));
    const errors=rows.filter(r=>r.true_label!==r.predicted_label).map(r=>r.image_id);
    assert.deepEqual(errors,wrongLatest.inputs[key].all_wrong_image_ids);
    assert.deepEqual(errors,wrong.filter(r=>r.architecture===a).map(r=>r.image_id));
    for(const r of wrong.filter(r=>r.architecture===a)){
      const saved=predictions[a].get(r.image_id);assert.equal(saved.predicted_label,r.predicted_label);assert.equal(saved.true_label,r.true_label);assert.equal(+saved.confidence,+r.confidence);
      assert.equal(hash(out+r.sample_directory+'/overlay.png'),wrongLatest.artifact_sha256[r.sample_directory+'/overlay.png']);
    }
  }
  assert.deepEqual([...predictions.resnet50.keys()],[...predictions.mobilenet_v3_large.keys()]);
  for(const r of fullErrorCases){const saved=predictions[r.architecture].get(r.image_id);assert.equal(saved.predicted_label,r.predicted_label);assert.equal(saved.true_label,r.true_label);assert.equal(+saved.confidence,+r.confidence);assert.equal(r.explained_target,r.predicted_label);assert.equal(r.checkpoint_sha256,counterparts.inputs[r.architecture+'_'+r.mode].checkpoint_sha256);}
  for(const r of correctRows)for(const [a,c]of [['resnet50',r.res],['mobilenet_v3_large',r.mob]]){const saved=predictions[a].get(r.id);assert.equal(saved.predicted_label,saved.true_label);assert.equal(saved.predicted_label,c.predicted_label);assert.equal(+saved.confidence,+c.confidence);}
  await lossFigure();
  await grid(correctRows,'Correct predictions · one example per class','gradcam_matched_correct_both_models_seed42.png');
  await grid(wrongRows,'All misclassified images · either model','gradcam_matched_all_wrong_both_models_seed42.png',true);
  const comparison=[...correctIds,...wrongIds].map((id,i)=>{
    const a=predictions.resnet50.get(id),b=predictions.mobilenet_v3_large.get(id);assert.equal(a.true_label,b.true_label);
    return {group:i<4?'both_correct':'either_wrong',image_id:id,true_label:a.true_label,resnet_pred:a.predicted_label,resnet_confidence:+a.confidence,mobile_pred:b.predicted_label,mobile_confidence:+b.confidence,resnet_overlay_shown:true,mobile_overlay_shown:true};
  });
  const keys=Object.keys(comparison[0]);fs.writeFileSync(path.join(__dirname,'matched_cam_readme_cases.csv'),[keys.join(','),...comparison.map(r=>keys.map(k=>r[k]).join(','))].join('\n')+'\n');
  assert.equal(placements.filter(r=>r.figure.endsWith('gradcam_matched_all_wrong_both_models_seed42.png')).length,14);
  assert.equal(placements.filter(r=>r.figure.endsWith('gradcam_matched_correct_both_models_seed42.png')).length,8);
  for(const p of protectedPaths)assert.equal(hash(p),before[p]);for(const [p,h]of Object.entries(inputs))assert.equal(hash(p),h);
  fs.writeFileSync(path.join(__dirname,'matched_cam_layout_sources.json'),JSON.stringify({created_utc:new Date().toISOString(),inputs_sha256:inputs,outputs_sha256:Object.fromEntries(outputs.map(p=>[p,hash(p)])),protected_sha256:before,seed:42,partition:'test',correct_image_ids:correctIds,wrong_image_ids:wrongIds,wrong_model_image_cases:9,error_group_overlay_count:14,correct_counterpart_overlay_count:5,display_policy:'Both overlays shown on every image; green labels correct predictions and red labels errors.',counterpart_diagnostics:{manifest:out+'gradcam/matched_error_counterparts_v1/seed42/latest.json',new_inference_cases:4,reused_CAM_cases:10,training_runs_added:0},comparison_csv:'report/matched_cam_readme_cases.csv',comparison,placements,python_inference_or_training_performed:false},null,2)+'\n');
  fs.writeFileSync(path.join(root,standalone+'loss_figure_sources.json'),JSON.stringify({created_utc:new Date().toISOString(),figure:standalone+'resnet_full_mobilenet_partial_loss_five_seeds.png',setups,mean_selected_epochs:[8.4,11.4],sources_sha256:Object.fromEntries(Object.entries(inputs).filter(([p])=>p.includes('history.csv')||p.includes('learning_curve_summary')||p.includes('final_checkpoint_summary')||p.includes('mean_selected_epochs'))),ddof:1,observed_epochs_only:true,python_inference_or_training_performed:false},null,2)+'\n');
  console.log(JSON.stringify({correct_rows:4,wrong_rows:7,error_group_overlays:14,error_cases:9,correct_counterparts:5,prediction_rows_verified:170,standalone_loss:standalone+'resnet_full_mobilenet_partial_loss_five_seeds.png'}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
