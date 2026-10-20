// Read-only recomputation of report evidence; writes one verification JSON.
// Does not invoke Python, model inference, or training.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),out=path.join(root,'outputs_modes_v3_td');
const study=path.join(out,'ablation','resnet50_full_v1');
const classes=['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia'],seeds=[42,123,456,789,1024];
const metrics=['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'];
const json=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const assert=(ok,message)=>{if(!ok)throw Error(message);};
function csv(file){const s=fs.readFileSync(file,'utf8').replace(/^\uFEFF/,'');let rows=[],row=[],f='',q=false;for(let i=0;i<s.length;i++){const c=s[i];if(q){if(c==='"'&&s[i+1]==='"'){f+='"';i++;}else if(c==='"')q=false;else f+=c;}else if(c==='"')q=true;else if(c===','){row.push(f);f='';}else if(c==='\n'){row.push(f.replace(/\r$/,''));rows.push(row);row=[];f='';}else f+=c;}if(f||row.length){row.push(f.replace(/\r$/,''));rows.push(row);}const h=rows.shift();return rows.filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])));}
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const sd=a=>Math.sqrt(a.reduce((s,x)=>s+(x-mean(a))**2,0)/(a.length-1));
let maxMetricError=0,maxSummaryError=0,maxStatisticError=0;
const compare=(a,b,kind)=>{const d=Math.abs(Number(a)-Number(b));assert(Number.isFinite(d)&&d<1e-9,`${kind} discrepancy: ${a}, ${b}`);return d;};
const source=json(path.join(__dirname,'resnet_followup_sources.json'));
const lock=json(path.join(study,'ablation_lock.json')),registry=json(path.join(study,'run_registry.json'));
assert(source.ablation.registry_sha256===hash(path.join(study,'run_registry.json')),'Registry changed after analysis');
assert(source.ablation.study_fingerprint===lock.study_fingerprint,'Study fingerprint mismatch');
const copies={'condition_summary.csv':'resnet_ablation_condition_summary.csv','per_class_summary.csv':'resnet_ablation_per_class_summary.csv','matched_deltas.csv':'resnet_ablation_matched_deltas.csv','statistical_tests.csv':'resnet_ablation_statistical_tests.csv','ablation_summary.png':'report_assets/latest/resnet_ablation_summary.png'};
for(const [name,digest]of Object.entries(source.ablation.artifact_sha256)){assert(hash(path.join(study,name))===digest,'Ablation source hash: '+name);assert(hash(path.join(__dirname,copies[name]))===digest,'Ablation copy hash: '+name);}
const camdir=path.join(out,source.gradcam.directory);
for(const [name,digest]of Object.entries(source.gradcam.artifact_sha256))assert(hash(path.join(camdir,name))===digest,'CAM source hash: '+name);
assert(hash(path.join(__dirname,'report_assets/latest/gradcam_resnet50_full_seed42.png'))===source.gradcam.artifact_sha256['gradcam_grid.png'],'CAM grid copy hash');
assert(hash(path.join(__dirname,'resnet_gradcam_cases.csv'))===source.gradcam.artifact_sha256['selected_cases.csv'],'CAM cases copy hash');
const original=json(path.join(out,'run_registry.json'));
assert(json(path.join(out,'selection.json')).selection_fingerprint===lock.original_selection_fingerprint,'Original selection lock differs from the ablation plan');
const test=csv(path.join(out,'splits.csv')).filter(r=>r.split==='test');
const groups={baseline:original.final_runs.filter(r=>r.architecture==='resnet50'&&r.mode==='full'&&r.status==='completed')};
for(const c of ['no_augmentation','no_class_weights'])groups[c]=registry.attempts.filter(r=>r.condition===c&&r.status==='completed');
let predictionRows=0;
const calculated={};
for(const [condition,runs]of Object.entries(groups)){
 assert(runs.length===5&&seeds.every(s=>runs.filter(r=>Number(r.seed)===s).length===1),'Five matched seeds missing: '+condition);
 calculated[condition]={};
 for(const r of runs){
  const cfg=json(path.join(r.run_dir,'config.json'));
  for(const [key,value]of Object.entries(lock.baseline_config)){const expected=lock.variants[condition]?.[key]??value;assert(JSON.stringify(cfg[key])===JSON.stringify(expected),'Config factor mismatch: '+condition+'/'+key);}
  const pred=csv(path.join(r.run_dir,'test_predictions.csv'));predictionRows+=pred.length;
  assert(pred.length===85&&pred.every((p,i)=>p.image_id===test[i].image_id&&p.true_label===test[i].label),'Canonical prediction order mismatch');
  const mat=classes.map(()=>classes.map(()=>0));for(const p of pred)mat[classes.indexOf(p.true_label)][classes.indexOf(p.predicted_label)]++;
  const pc=classes.map((c,i)=>{const tp=mat[i][i],support=mat[i].reduce((a,b)=>a+b,0),col=mat.reduce((n,row)=>n+row[i],0),p=col?tp/col:0,rec=tp/support;return {class:c,precision:p,recall:rec,f1:p+rec?2*p*rec/(p+rec):0,support};});
  const vals={test_acc:mat.reduce((n,row,i)=>n+row[i],0)/85,test_macro_precision:mean(pc.map(v=>v.precision)),test_macro_recall:mean(pc.map(v=>v.recall)),test_macro_f1:mean(pc.map(v=>v.f1)),pc};
  for(const m of metrics)maxMetricError=Math.max(maxMetricError,compare(vals[m],r[m],'run metric'));
  calculated[condition][r.seed]=vals;
 }
}
for(const row of csv(path.join(study,'condition_summary.csv')))for(const m of metrics){const v=seeds.map(s=>calculated[row.condition][s][m]);maxSummaryError=Math.max(maxSummaryError,compare(mean(v),row[m+'_mean'],'mean'),compare(sd(v),row[m+'_sample_sd'],'sample SD'));}
for(const row of csv(path.join(study,'matched_deltas.csv')))for(const m of metrics)maxSummaryError=Math.max(maxSummaryError,compare(100*(calculated[row.condition][row.seed][m]-calculated.baseline[row.seed][m]),row[m+'_delta_pp'],'matched delta'));
for(const row of csv(path.join(study,'per_class_summary.csv'))){for(const m of ['precision','recall','f1']){const v=seeds.map(s=>calculated[row.condition][s].pc.find(p=>p.class===row.class)[m]);maxSummaryError=Math.max(maxSummaryError,compare(mean(v),row[m+'_mean'],'per-class mean'),compare(sd(v),row[m+'_sample_sd'],'per-class SD'));}assert(Number(row.support)===calculated[row.condition][42].pc.find(p=>p.class===row.class).support,'Per-class support mismatch');}
const tests=csv(path.join(study,'statistical_tests.csv'));assert(tests.length===4,'Expected four tests');
for(const row of tests){const diff=seeds.map(s=>calculated[row.condition][s][row.metric]-calculated.baseline[s][row.metric]),t=mean(diff)/(sd(diff)/Math.sqrt(5)),u=Math.abs(t)/Math.sqrt(t*t+4),p=.5*(1-u)**2*(u+2);maxStatisticError=Math.max(maxStatisticError,compare(mean(diff),row.mean_difference,'paired mean'),compare(t,row.t_statistic,'paired t'),compare(p,row.p_value,'paired p'));assert(row.status==='ok'&&Number(row.holm_family_size)===4&&Number(row.paired_n)===5,'Statistical family mismatch');}
let running=0;[...tests].sort((a,b)=>Number(a.p_value)-Number(b.p_value)).forEach((r,i)=>{running=Math.max(running,Number(r.p_value)*(4-i));maxStatisticError=Math.max(maxStatisticError,compare(Math.min(1,running),r.p_holm,'Holm p'));assert((Number(r.p_holm)<.05)===(r.significant_holm==='True'),'Significance flag mismatch');});
const cases=csv(path.join(camdir,'selected_cases.csv'));
assert(cases.length===8&&cases.every(r=>r.architecture==='resnet50'&&r.mode==='full'&&Number(r.seed)===42),'CAM identities mismatch');
const seed42=csv(path.join(groups.baseline.find(r=>r.seed===42).run_dir,'test_predictions.csv'));
for(const c of cases){const p=seed42.find(p=>p.image_id===c.image_id);assert(p&&p.predicted_label===c.predicted_label&&p.true_label===c.true_label&&p.confidence===c.confidence,'CAM case mismatch');}
const mainRuns=[...original.trials,...original.final_runs].filter(r=>r.status==='completed');
const newRuns=registry.attempts.filter(r=>r.status==='completed');
const result={verified_utc:new Date().toISOString(),method:'Node.js recomputation from saved prediction files; no Python/inference/training',baseline_reused:5,new_ablation_runs:10,unique_new_seeds_per_condition:5,matched_prediction_rows:predictionRows,cam_examples:cases.length,cam_correct:cases.filter(c=>c.true_label===c.predicted_label).length,cam_wrong:cases.filter(c=>c.true_label!==c.predicted_label).length,completed_ablation_epochs:newRuns.reduce((n,r)=>n+r.epochs_completed,0),completed_main_runs:mainRuns.length,total_completed_training_runs:mainRuns.length+newRuns.length,total_completed_training_epochs:[...mainRuns,...newRuns].reduce((n,r)=>n+r.epochs_completed,0),max_run_metric_error:maxMetricError,max_summary_error:maxSummaryError,max_statistics_error:maxStatisticError,significant_holm_tests:tests.filter(r=>r.significant_holm==='True').length,artifact_hashes_match:true,original_selection_matches_ablation_lock:true,effective_tracking_modes:[...new Set(registry.attempts.map(r=>r.tracking_mode))],cloud_completeness_verified:false};
fs.writeFileSync(path.join(__dirname,'resnet_followup_verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
