// Read-only numerical/layout checks for the report; no model code, Python or training.
const fs=require('fs'),path=require('path');
const {root,readCsv,hash,mean,sd}=require('./report_utils');
const old=JSON.parse(fs.readFileSync(path.join(__dirname,'qa/report_revision_inputs.json'),'utf8'));
for(const [p,h] of Object.entries(old.source_sha256))if(hash(p)!==h)throw Error('Measured artifact changed: '+p);
const summary=readCsv('report/learning_curve_summary.csv'), registry=JSON.parse(fs.readFileSync(path.join(root,'outputs_modes_v3_td/run_registry.json'),'utf8'));
const runs=registry.final_runs.filter(r=>r.status==='completed').map(r=>({...r,history:readCsv(path.relative(root,path.join(r.run_dir,'history.csv'))).filter(h=>h.train_loss!=='')}));
let maxMeanError=0,maxSdError=0,singleRunEpochMetrics=0;
for(const row of summary){
  const observations=runs.filter(r=>r.architecture===row.architecture&&r.mode===row.mode).flatMap(r=>r.history.filter(h=>Number(h.epoch)===Number(row.epoch)).map(h=>Number(h[row.metric])));
  if(observations.length!==Number(row.n)||!observations.length)throw Error('Incorrect available-run count');
  maxMeanError=Math.max(maxMeanError,Math.abs(mean(observations)-Number(row.mean)));
  if(observations.length===1){singleRunEpochMetrics++;if(row.sample_sd!=='')throw Error('Sample SD present for one run');}
  else maxSdError=Math.max(maxSdError,Math.abs(sd(observations)-Number(row.sample_sd)));
}
if(maxMeanError>1e-12||maxSdError>1e-12)throw Error('Learning aggregation mismatch');
const hist=readCsv('report/image_property_histograms.csv');let histogramGroups=0;
for(const figure of ['aspect_ratio','brightness'])for(const c of ['san_phraphrom','san_phraphum','san_tayay','tee_ju_eia']){
  const group=hist.filter(r=>r.figure===figure&&r.class===c);if(!group.length)throw Error('Missing histogram group');
  const n=group.reduce((a,r)=>a+Number(r.count),0),pct=group.reduce((a,r)=>a+Number(r.proportion_pct),0);
  if(n!==Number(group[0].class_total)||Math.abs(pct-100)>1e-9)throw Error('Histogram omitted images or denominator mismatch');histogramGroups++;
}
const tuning=readCsv('report/tuning_checkpoint_observations.csv');if(tuning.length!==75||tuning.filter(r=>r.winner==='true').length!==15)throw Error('Incorrect tuning winner count');
const checkpoints=readCsv('report/final_checkpoint_summary.csv');if(checkpoints.length!==75)throw Error('Missing final checkpoint identities');
const readme=fs.readFileSync(path.join(root,'README.md'),'utf8');const lines=readme.split(/\r?\n/);
if(lines.some(l=>/^#+ /.test(l)&&/[ก-๙]/.test(l)))throw Error('Non-English heading');
let inTable=false,tables=0;for(const l of lines){if(l.startsWith('|')){if(!inTable){tables++;if(/[ก-๙]/.test(l))throw Error('Non-English table header');}inTable=true;}else inTable=false;}
if(/เรา|โจทย์|ผลการทดลองจาก.*ปรับปรุง|ปิด augmentation/.test(readme))throw Error('Outdated report language');
const mirror=fs.readFileSync(path.join(__dirname,'report_source.md'),'utf8').split('\n').slice(1).join('\n');
if(mirror!==readme.replaceAll('](report/',']('))throw Error('Report mirror differs');
const enhanced=JSON.parse(fs.readFileSync(path.join(__dirname,'enhanced_figure_sources.json'),'utf8'));
for(const [p,h]of Object.entries(enhanced.source_sha256))if(hash(p)!==h)throw Error('Figure input drift');
for(const [p,h]of Object.entries(enhanced.output_sha256))if(hash(p)!==h)throw Error('Figure output drift');
const evidence={checked_utc:new Date().toISOString(),unchanged_measured_csv_json_files:Object.keys(old.source_sha256).length,
  learning_curve_metric_rows:summary.length,final_runs:75,final_pairs:15,history_mean_max_error:maxMeanError,history_sample_sd_max_error:maxSdError,
  single_run_epoch_metric_rows_without_sd:singleRunEpochMetrics,histogram_groups_with_all_images_and_100pct:histogramGroups,
  tuning_trials:75,within_setup_winners:15,final_checkpoint_identities:75,english_headings_and_table_headers:true,personal_pronouns_removed:true,table_count:tables,
  selection_lock_unchanged:true,ablation_lock_unchanged:true,python_inference_training_performed:false};
fs.writeFileSync(path.join(__dirname,'qa/report_revision_validation.json'),JSON.stringify(evidence,null,2)+'\n');
console.log(JSON.stringify(evidence));
