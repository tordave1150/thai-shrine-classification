// Report-only evidence checks. Does not import Python, train, or run inference.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {root,readCsv,hash,mean,sd}=require('./report_utils');
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const evidence=read('report/readme_revision_evidence.json');
const curves=read('report/enhanced_figure_sources.json');
const preservation=read('report/qa/evidence_preservation.json');
const previousFollowup=read('report/resnet_followup_sources.json');
const verifiedSources={};
for(const manifest of [evidence,curves])for(const [file,digest] of Object.entries(manifest.source_sha256)){
  assert.equal(hash(file),digest,'Saved source changed: '+file);verifiedSources[file]=digest;
}
for(const manifest of [evidence,curves])for(const [file,digest] of Object.entries(manifest.output_sha256))assert.equal(hash(file),digest,'Figure changed: '+file);
const base='outputs_modes_v3_td/';
const ablation=base+'ablation/resnet50_full_v1/';
assert.equal(hash(base+'selection.json'),preservation.selection_sha256);
assert.equal(hash(ablation+'ablation_lock.json'),preservation.ablation_lock_sha256);
assert.equal(hash(ablation+'run_registry.json'),previousFollowup.ablation.registry_sha256);
for(const [file,digest] of Object.entries(previousFollowup.ablation.artifact_sha256))assert.equal(hash(ablation+file),digest);
const selected=read(base+'selection.json');
assert.deepEqual(selected.reference_pair,{architecture:'mobilenet_v3_large',mode:'partial'});
const summary=readCsv(base+'final/pair_summary.csv');assert.equal(summary.length,15);
for(const key of ['test_acc','test_macro_precision','test_macro_recall','test_macro_f1']){
  const top=[...summary].sort((a,b)=>Number(b[key+'_mean'])-Number(a[key+'_mean']))[0];
  assert.equal(top.architecture+'::'+top.mode,'resnet50::full');
  assert.equal(Number(top[key+'_mean']),evidence.leaders[key].mean);
}
const checkpoints=readCsv('report/final_checkpoint_summary.csv');assert.equal(checkpoints.length,75);
const epochMeans=readCsv('report/mean_selected_epochs.csv');assert.equal(epochMeans.length,15);
for(const r of epochMeans){
  const group=checkpoints.filter(x=>x.architecture===r.architecture&&x.mode===r.mode);
  assert.equal(group.length,5);assert.equal(Number(r.mean_selected_epoch),mean(group.map(x=>Number(x.selected_epoch))));
  const svg=fs.readFileSync(path.join(root,'report/report_assets/latest/learning_curves_'+r.architecture+'.svg'),'utf8');
  assert.equal((svg.match(/Mean selected epoch = /g)||[]).length,3);
  assert.equal((svg.match(/stroke-dasharray="3 5"/g)||[]).length,3);
  assert(svg.includes('Mean selected epoch = '+Number(r.mean_selected_epoch).toFixed(1)));
}
const runs=readCsv(base+'final/run_metrics.csv');assert.equal(runs.length,75);
const differences=evidence.paired_f1_example.matched.map(r=>{
  const metric=(a,m)=>Number(runs.find(x=>Number(x.seed)===r.seed&&x.architecture===a&&x.mode===m).test_macro_f1)*100;
  assert.equal(metric('mobilenet_v3_large','partial'),r.mobile_f1_pct);
  assert.equal(metric('resnet50','full'),r.resnet_f1_pct);
  return r.mobile_f1_pct-r.resnet_f1_pct;
});
const example=evidence.paired_f1_example;
assert.equal(mean(differences),example.mean_difference_pp);assert.equal(sd(differences),example.sample_sd_difference_pp);
const t=mean(differences)/(sd(differences)/Math.sqrt(5));assert.equal(t,example.t);
// Exact two-sided t probability for df=4, independent of the saved p-value.
const u=Math.abs(t)/Math.sqrt(t*t+4),p=1-1.5*u+.5*u*u*u;
assert(Math.abs(p-example.p_raw)<1e-12);
const stats=readCsv('report/verified_statistics.csv');assert.equal(stats.length,58);
const primary=stats.filter(r=>r.family_id.startsWith('primary'));assert.equal(primary.length,28);
assert.equal(stats.filter(r=>r.family_id.startsWith('secondary')).length,30);
let adjusted=0;
for(const [i,r]of [...primary].sort((a,b)=>Number(a.p_raw)-Number(b.p_raw)).entries()){
  adjusted=Math.min(1,Math.max(adjusted,(28-i)*Number(r.p_raw)));
  if(r.competitor==='resnet50::full'&&r.metric==='test_macro_f1')assert.equal(adjusted,example.p_holm);
}
const bins=readCsv(base+'confidence/bin_metrics.csv');
const ece={};
for(const [arch,mode]of [['resnet50','full'],['mobilenet_v3_large','partial']]){
  const rows=bins.filter(r=>r.architecture===arch&&r.mode===mode&&r.seed==='42');
  assert.equal(rows.length,10);assert.equal(rows.reduce((s,r)=>s+Number(r.count),0),85);
  ece[arch]=100*rows.reduce((s,r)=>s+(Number(r.count)?Number(r.count)/85*Math.abs(Number(r.mean_confidence)-Number(r.accuracy)):0),0);
}
assert.equal(ece.resnet50.toFixed(2),'5.99');assert.equal(ece.mobilenet_v3_large.toFixed(2),'2.76');
const latest=read(base+'gradcam/mobilenet_v3_large_partial_matched_resnet/seed42/latest.json');
assert.equal(latest.examples,8);assert.equal(latest.training_runs_added,0);
for(const [file,digest]of Object.entries(latest.artifact_sha256))assert.equal(hash(base+file),digest);
const mobile=readCsv(base+latest.directory+'/selected_cases.csv');
const resnet=readCsv('report/resnet_gradcam_cases.csv');
assert.deepEqual(mobile.map(r=>r.image_id),resnet.map(r=>r.image_id));
assert.equal(new Set(mobile.map(r=>r.image_id)).size,8);
const classes=['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia'];
for(const r of mobile){
  const probabilities=JSON.parse(r.cam_inference_probabilities);
  assert.equal(classes[probabilities.indexOf(Math.max(...probabilities))],r.predicted_label);
}
const markdown=fs.readFileSync(path.join(root,'README.md'),'utf8');
assert(!/1\.4 Duplicates|7\.2 Future work|ต้องประเมินยืนยันด้วยข้อมูลภายนอกชุดใหม่|\{\{/.test(markdown));
assert.equal((markdown.match(/### 6\.9 Final model selection/g)||[]).length,1);
assert(markdown.indexOf('### 6.9 Final model selection')>markdown.indexOf('### 6.8 Confidence and calibration'));
assert(markdown.includes('Holm-adjusted p = 1.0000 > α = 0.05'));
assert(markdown.includes('0.6514')&&markdown.includes('4.0327'));
const result={checked_utc:new Date().toISOString(),readme_sha256:hash('README.md'),template_sha256:hash('report/readme_template.md'),
  source_hashes_checked:Object.keys(verifiedSources).length,main_registry_sha256:hash(base+'run_registry.json'),
  selection_sha256:hash(base+'selection.json'),ablation_lock_sha256:hash(ablation+'ablation_lock.json'),
  preserved_main_final_runs:75,preserved_statistical_families:[28,30],mean_epoch_lines:15,
  metric_leaders:evidence.leaders,paired_f1:{t,raw_p_recomputed:p,holm_p:example.p_holm},seed42_ece_pct:ece,
  matched_cam_images:8,cam_prediction_classes_match_saved:true,
  maximum_cam_probability_difference:Math.max(...mobile.map(r=>Number(r.probability_max_abs_error))),
  execution_scope:'Report-only Node verification; eight unique MobileNet Grad-CAM images were generated under separate explicit authorization. No new training/tuning.',passed:true};
fs.writeFileSync(path.join(__dirname,'qa/readme_revision_checks.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
