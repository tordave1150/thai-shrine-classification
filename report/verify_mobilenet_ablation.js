// Independent recomputation from saved MobileNet predictions. No model execution.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {root,readCsv,hash,mean,sd}=require('./report_utils');
const out='outputs_modes_v3_td/',read=p=>fs.readFileSync(path.join(root,p),'utf8').replace(/^\uFEFF/,''),json=p=>JSON.parse(read(p));
const source=json('report/mobilenet_ablation_sources.json'),study=out+source.study_directory+'/';
const lock=json(study+'ablation_lock.json'),registry=json(study+'run_registry.json'),original=json(out+'run_registry.json');
const seeds=[42,123,456,789,1024],classes=['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia'];
const metrics=['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'];
const baseline=original.final_runs.filter(r=>r.status==='completed'&&r.architecture==='mobilenet_v3_large'&&r.mode==='partial');
const groups={baseline,no_augmentation:registry.attempts.filter(r=>r.condition==='no_augmentation'&&r.status==='completed'),no_class_weights:registry.attempts.filter(r=>r.condition==='no_class_weights'&&r.status==='completed')};
const testRows=readCsv(out+'splits.csv').filter(r=>r.split==='test'),calculated={},max={metric:0,summary:0,statistic:0};let predictionRows=0;
const rel=p=>{const value=String(p).replaceAll('\\','/'),token='/outputs_modes_v3_td/';return value.includes(token)?'outputs_modes_v3_td/'+value.split(token)[1]:path.relative(root,p).replaceAll('\\','/');};
const compare=(a,b,type)=>{const d=Math.abs(Number(a)-Number(b));assert(Number.isFinite(d)&&d<1e-9,type+': '+a+' versus '+b);max[type]=Math.max(max[type],d);};
assert.equal(lock.architecture,'mobilenet_v3_large');assert.equal(lock.mode,'partial');assert.equal(lock.statistics.family_size,4);
assert.equal(source.analysis.study_fingerprint,lock.study_fingerprint);assert.equal(source.analysis.registry_sha256,hash(study+'run_registry.json'));
assert.equal(lock.original_selection_fingerprint,json(out+'selection.json').selection_fingerprint);assert.equal(lock.split_sha256,hash(out+'splits.csv'));
for(const [p,h]of Object.entries(lock.baseline_artifact_sha256))assert.equal(hash(out+p),h,p);
for(const [p,h]of Object.entries(source.analysis.artifact_sha256))assert.equal(hash(study+p),h,p);
for(const p of ['condition_summary.csv','per_class_summary.csv','matched_deltas.csv','statistical_tests.csv'])assert.equal(hash(study+p),hash('report/mobilenet_ablation_'+p));
assert.equal(hash(study+'ablation_summary.png'),hash('report/report_assets/latest/mobilenet_ablation_summary.png'));
for(const [condition,runs]of Object.entries(groups)){
 assert.equal(runs.length,5);assert.deepEqual(runs.map(r=>r.seed).sort((a,b)=>a-b),seeds);calculated[condition]={};
 for(const r of runs){
  const dir=rel(r.run_dir)+'/',cfg=json(dir+'config.json');assert.equal(cfg.architecture,'mobilenet_v3_large');assert.equal(cfg.mode,'partial');assert.equal(cfg.pretrained,true);assert.equal(cfg.weights,'MobileNet_V3_Large_Weights.DEFAULT');
  assert.equal(cfg.mode_stats.trainable_parameters,3414668);assert.deepEqual(cfg.mode_stats.trainable_module_names,['features.13','features.14','features.15','features.16','classifier']);
  for(const [k,v]of Object.entries(lock.baseline_config))assert.deepEqual(cfg[k],lock.variants[condition]?.[k]??v,'factor mismatch: '+condition+'/'+k);
  const predictions=readCsv(dir+'test_predictions.csv');predictionRows+=predictions.length;assert.equal(predictions.length,85);
  const matrix=classes.map(()=>classes.map(()=>0));
  predictions.forEach((p,i)=>{assert.equal(p.image_id,testRows[i].image_id);assert.equal(p.true_label,testRows[i].label);assert(classes.includes(p.predicted_label));const probabilities=classes.map(c=>+p['prob_'+c]);assert(probabilities.every(v=>Number.isFinite(v)&&v>=0&&v<=1));assert(Math.abs(probabilities.reduce((a,b)=>a+b,0)-1)<1e-5);assert.equal(classes[probabilities.indexOf(Math.max(...probabilities))],p.predicted_label);assert(Math.abs(+p.confidence-Math.max(...probabilities))<1e-6);matrix[classes.indexOf(p.true_label)][classes.indexOf(p.predicted_label)]++;});
  const perClass=classes.map((c,i)=>{const tp=matrix[i][i],support=matrix[i].reduce((a,b)=>a+b,0),column=matrix.reduce((n,row)=>n+row[i],0),precision=column?tp/column:0,recall=tp/support;return {class:c,support,precision,recall,f1:precision+recall?2*precision*recall/(precision+recall):0};});
  const values={test_acc:matrix.reduce((n,row,i)=>n+row[i],0)/85,test_macro_precision:mean(perClass.map(r=>r.precision)),test_macro_recall:mean(perClass.map(r=>r.recall)),test_macro_f1:mean(perClass.map(r=>r.f1)),perClass};
  for(const m of metrics)compare(values[m],r[m],'metric');calculated[condition][r.seed]=values;
  const history=readCsv(dir+'history.csv').filter(r=>r.train_loss!=='');assert.equal(history.length,r.epochs_completed);history.forEach((h,i)=>assert.equal(+h.epoch,i+1));
  const best=[...history].sort((a,b)=>(+b.val_macro_f1-+a.val_macro_f1)||(+a.val_loss-+b.val_loss)||(+a.epoch-+b.epoch))[0];assert.equal(+best.epoch,r.best_epoch);
  if(condition!=='baseline'){for(const [p,h]of Object.entries(r.artifact_sha256))assert.equal(hash(study+p),h,p);for(const image of ['curves.png','confusion_matrix.png','confusion_matrix_normalized.png'])assert(fs.existsSync(path.join(root,dir+image)),image);}
 }
}
for(const row of readCsv(study+'condition_summary.csv'))for(const m of metrics){const values=seeds.map(s=>calculated[row.condition][s][m]);compare(mean(values),row[m+'_mean'],'summary');compare(sd(values),row[m+'_sample_sd'],'summary');}
for(const row of readCsv(study+'matched_deltas.csv'))for(const m of metrics)compare(100*(calculated[row.condition][row.seed][m]-calculated.baseline[row.seed][m]),row[m+'_delta_pp'],'summary');
for(const row of readCsv(study+'per_class_summary.csv')){for(const m of ['precision','recall','f1']){const values=seeds.map(s=>calculated[row.condition][s].perClass.find(r=>r.class===row.class)[m]);compare(mean(values),row[m+'_mean'],'summary');compare(sd(values),row[m+'_sample_sd'],'summary');}assert.equal(+row.support,calculated[row.condition][42].perClass.find(r=>r.class===row.class).support);}
const tests=readCsv(study+'statistical_tests.csv');assert.equal(tests.length,4);
for(const row of tests){assert.equal(row.family_id,lock.statistics.family_id);assert.equal(+row.holm_family_size,4);assert.equal(+row.paired_n,5);const diff=seeds.map(s=>calculated[row.condition][s][row.metric]-calculated.baseline[s][row.metric]);compare(mean(diff),row.mean_difference,'statistic');if(sd(diff)===0){assert.equal(row.status,'undefined_zero_variance');continue;}const t=mean(diff)/(sd(diff)/Math.sqrt(5)),u=Math.abs(t)/Math.sqrt(t*t+4),p=.5*(1-u)**2*(u+2);assert.equal(row.status,'ok');compare(t,row.t_statistic,'statistic');compare(p,row.p_value,'statistic');}
let running=0;tests.filter(r=>r.status==='ok').sort((a,b)=>a.p_value-b.p_value).forEach((r,i)=>{running=Math.max(running,+r.p_value*(4-i));compare(Math.min(1,running),r.p_holm,'statistic');assert.equal(+r.p_holm<.05,String(r.significant_holm).toLowerCase()==='true');});
const fresh=[...groups.no_augmentation,...groups.no_class_weights],previous=json(out+'ablation/resnet50_full_v1/run_registry.json').attempts.filter(r=>r.status==='completed');
const initial=json('report/mobilenet_ablation_pre_execution.json');
for(const [p,h]of Object.entries(initial.sha256))if(p!=='README.md')assert.equal(hash(p),h,p+' changed');
const result={verified_utc:new Date().toISOString(),method:'Independent Node.js recomputation from canonical predictions, matched-seed differences, df=4 paired t probabilities and fixed-family Holm; no model execution by this verifier.',architecture:'mobilenet_v3_large',mode:'partial',study_directory:source.study_directory,baseline_reused:5,new_completed_runs:10,new_completed_epochs:fresh.reduce((n,r)=>n+r.epochs_completed,0),prediction_rows_checked:predictionRows,paired_seeds:seeds,holm_family_size:4,significant_holm_tests:tests.filter(r=>String(r.significant_holm).toLowerCase()==='true').length,max_run_metric_error:max.metric,max_summary_error:max.summary,max_statistic_error:max.statistic,source_and_copied_artifact_hashes_match:true,all_training_curve_and_confusion_images_present:true,original_main_and_resnet_evidence_unchanged:true,initial_failed_directory_attempts:10,initial_failed_attempt_epochs:0,total_completed_training_runs:170,total_completed_training_epochs:[...original.trials,...original.final_runs,...previous,...fresh].filter(r=>r.status==='completed').reduce((n,r)=>n+r.epochs_completed,0),effective_tracking_modes:[...new Set(fresh.map(r=>r.tracking_mode))],wandb_run_urls:fresh.map(r=>r.wandb_run_url),cloud_dashboard_completeness_verified:false,source_sha256:{ablation_lock:hash(study+'ablation_lock.json'),ablation_registry:hash(study+'run_registry.json'),original_selection:hash(out+'selection.json'),original_registry:hash(out+'run_registry.json')},errors:[]};
fs.writeFileSync(path.join(root,'report/mobilenet_ablation_verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
