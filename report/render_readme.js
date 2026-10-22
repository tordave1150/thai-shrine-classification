// Report-only renderer. All percentages are display conversions of saved 0–1 metrics.
// Run: node report/render_readme.js. This never imports experiment code.
const fs = require('fs');
const path = require('path');
const { root, readCsv, hash } = require('./report_utils');
const names = {resnet50:'ResNet50', efficientnet_b0:'EfficientNet-B0', mobilenet_v3_large:'MobileNetV3-Large', densenet121:'DenseNet121', convnext_tiny:'ConvNeXt-Tiny'};
const modes = {full:'Full', frozen:'Frozen', partial:'Partial'};
const classes = {san_phraphrom:'พระพรหม', san_phraphum:'พระภูมิ', san_tayay:'ตายาย', tee_ju_eia:'ตี่จู้เอี๊ยะ'};
const setup = r => names[r.architecture] + '/' + modes[r.mode];
const pairName = id => { const [architecture, mode] = id.split('::'); return setup({architecture, mode}); };
const f = (n, d=2) => Number(n).toFixed(d);
const pct = n => f(Number(n) * 100);
const pm = (r, key) => pct(r[key+'_mean']) + ' ± ' + pct(r[key+'_sample_sd']);
const pval = n => Number(n) < 0.001 ? '<0.001' : f(n, 3);
const inputs = ['report/readme_template.md', 'report/verified_pair_summary.csv', 'report/verified_per_class_summary.csv',
  'report/verified_statistics.csv', 'report/verified_training_dynamics.csv', 'report/resnet_gradcam_cases.csv',
  'report/resnet_ablation_condition_summary.csv', 'report/resnet_ablation_statistical_tests.csv',
  'outputs_modes_v3_td/tuning/best_params_by_arch_mode.json', 'outputs_modes_v3_td/tuning/search_definition.json',
  'outputs_modes_v3_td/eyeball/eyeball_cases.csv', 'report/error_frequency.csv', 'report/final_checkpoint_summary.csv',
  'report/readme_revision_evidence.json', 'report/matched_cam_comparison.csv'];
const before = Object.fromEntries(inputs.map(p => [p, hash(p)]));
function table(headers, rows, numericColumns=[]) {
  if (rows.some(r => r.length !== headers.length)) throw Error('Markdown table column mismatch');
  return ['| '+headers.join(' | ')+' |', '| '+headers.map((_, i) => numericColumns.includes(i)?'---:':'---').join(' | ')+' |',
    ...rows.map(r => '| '+r.map(v => String(v).replaceAll('|', '\\|')).join(' | ')+' |')].join('\n');
}
const blocks = {};
const revision = JSON.parse(fs.readFileSync(path.join(root,'report/readme_revision_evidence.json'),'utf8'));
blocks.HYPOTHESIS_F1_TABLE = table(['Seed','MobileNet/Partial F1 (%)','ResNet/Full F1 (%)','Difference (pp)'],
  revision.paired_f1_example.matched.map(r=>[r.seed,f(r.mobile_f1_pct,4),f(r.resnet_f1_pct,4),f(r.difference_pp,4)]),[0,1,2,3]);
blocks.MATCHED_CAM_TABLE = table(['Image ID (stem)','True class','ResNet prediction','Confidence (%)','MobileNet prediction','Confidence (%)'],
  readCsv('report/matched_cam_comparison.csv').map(r=>[r.image_id.split('/')[1],classes[r.true_label],
    classes[r.resnet_pred]+' '+(r.resnet_pred===r.true_label?'✓':'✗'),pct(r.resnet_confidence),
    classes[r.mobile_pred]+' '+(r.mobile_pred===r.true_label?'✓':'✗'),pct(r.mobile_confidence)]),[3,5]);
const summary = readCsv('report/verified_pair_summary.csv');
if (summary.length !== 15 || summary.some(r => Number(r.n_runs || r.completed_seeds || r.n_seeds || 5) !== 5)) throw Error('Expected 15 five-seed summaries');
blocks.RESULT_TABLE = table(['Setup','Accuracy (%)','Macro precision (%)','Macro recall (%)','Macro F1 (%)'],
  [...summary].sort((a,b)=>Number(b.test_macro_f1_mean)-Number(a.test_macro_f1_mean)).map(r => [setup(r), ...['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'].map(key => pm(r,key))]), [1,2,3,4]);
const classSummary = readCsv('report/verified_per_class_summary.csv');
const selectedClasses = classSummary.filter(r => (r.architecture==='resnet50'&&r.mode==='full') || (r.architecture==='mobilenet_v3_large'&&r.mode==='partial'));
if (selectedClasses.length !== 8) throw Error('Missing selected/reference per-class metrics');
blocks.PERCLASS_TABLE = table(['Setup','Class','Support','Precision (%)','Recall (%)','F1 (%)'], selectedClasses.map(r => [setup(r), classes[r.class], r.support, ...['precision','recall','f1'].map(key=>pm(r,key))]), [2,3,4,5]);
const stats = readCsv('report/verified_statistics.csv');
if(stats.length!==58)throw Error('Expected all 58 locked statistical comparisons');
const shownStats = stats.filter(r => r.significant.toLowerCase()==='true' || (r.reference==='mobilenet_v3_large::partial' && r.competitor==='resnet50::full'));
blocks.STATS_TABLE = table(['Family','A − B','Metric','Mean Δ (pp)','95% CI (pp)¹','Holm p'], shownStats.map(r => [r.family_id.startsWith('primary')?'Primary':'Secondary', pairName(r.reference)+'<br>− '+pairName(r.competitor), r.metric==='test_acc'?'Accuracy':'Macro F1', pct(r.mean_difference), '['+pct(r.ci95_lower)+', '+pct(r.ci95_upper)+']', pval(r.p_holm)]), [3,5]);
blocks.DYNAMICS_TABLE = table(['Setup','Mean selected epoch','Mean stop epoch','Training F1 (%)','Validation F1 (%)','Late loss divergence (runs)'], readCsv('report/verified_training_dynamics.csv').map(r => [setup(r),f(r.mean_best_epoch,1),f(r.mean_stop_epoch,1),pct(r.mean_train_f1_at_best),pct(r.mean_val_f1_at_best),r.late_divergence_runs+'/5']), [1,2,3,4,5]);
const winners = Object.values(JSON.parse(fs.readFileSync(path.join(root, inputs[8]),'utf8')).pairs);
if(winners.length!==15)throw Error('Expected all 15 independent tuning winners');
blocks.WINNER_TABLE = table(['Setup','Learning rate','Weight decay','Optimizer','Batch size','Tuning validation F1 (%)'], winners.map(r=>[setup(r),Number(r.learning_rate).toExponential(6),Number(r.weight_decay).toExponential(6),r.optimizer==='adamw'?'AdamW':r.optimizer==='sgd'?'SGD':'Adam',r.batch_size,pct(r.best_val_macro_f1)]), [1,2,4,5]);
const checkpointRows=readCsv('report/final_checkpoint_summary.csv');
if(checkpointRows.length!==75)throw Error('Expected 75 final checkpoint identities');
blocks.CHECKPOINT_SUMMARY_TABLE=table(['Setup','Final runs','Mean selected epoch','Selected epoch range','Seeds'],winners.map(w=>{const rows=checkpointRows.filter(r=>r.architecture===w.architecture&&r.mode===w.mode);if(rows.length!==5)throw Error('Missing checkpoint seeds');const epochs=rows.map(r=>Number(r.selected_epoch));return [setup(w),5,f(epochs.reduce((a,b)=>a+b,0)/5,1),Math.min(...epochs)+'–'+Math.max(...epochs),rows.map(r=>r.seed).join(', ')];}),[1,2]);
const search = JSON.parse(fs.readFileSync(path.join(root,inputs[9]),'utf8'));
blocks.SEARCH_TABLE = table(['Trial','Learning rate','Weight decay','Optimizer','Batch'], search.sampled_configs.map((r,i)=>[i+1,Number(r.learning_rate).toExponential(6),Number(r.weight_decay).toExponential(6),r.optimizer==='adamw'?'AdamW':'Adam',r.batch_size]), [0,1,2,4]);
const cases = readCsv('report/resnet_gradcam_cases.csv');
blocks.RESNET_CASE_TABLE = table(['Image ID (stem)','True class','Predicted class','Confidence (%)','Selection reason'], cases.map(r=>[r.image_id.split('/')[1],classes[r.true_label],classes[r.predicted_label]+' '+(r.true_label===r.predicted_label?'✓':'✗'),pct(r.confidence),r.selection_reason==='highest_confidence_correct'?'ถูก มั่นใจสูงสุดในคลาส':r.selection_reason==='highest_confidence_misclassified'?'ผิด มั่นใจสูงสุดในคลาส':'ถูก มั่นใจต่ำสุด; คลาสนี้ไม่มีภาพผิด']), [3]);
blocks.CASE_TABLE = table(['Image ID (stem)','True class','Fine-tuned prediction','Confidence (%)','ImageNet top-1'],readCsv(inputs[10]).map(r=>[r.image_id.split('/')[1],classes[r.true_label],classes[r.predicted_label]+' '+(r.true_label===r.predicted_label?'✓':'✗'),pct(r.confidence),r.imagenet_label]),[3]);
blocks.ERROR_TABLE = table(['Image ID (stem)','True class','Errors / 75','Reference errors / 5','ResNet Full errors / 5'],readCsv('report/error_frequency.csv').slice(0,10).map(r=>[r.image_id.split('/')[1],classes[r.true_label],r.wrong_all75,r.wrong_reference5,r.wrong_resnetfull5]),[2,3,4]);
const ablation = readCsv('report/resnet_ablation_condition_summary.csv');
const conditionNames = {baseline:'Baseline',no_augmentation:'ไม่ใช้ augmentation',no_class_weights:'ไม่ใช้ class weights'};
blocks.ABLATION_TABLE = table(['Condition','Accuracy (%)','Macro precision (%)','Macro recall (%)','Macro F1 (%)'],ablation.map(r=>[conditionNames[r.condition],...['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'].map(key=>pm(r,key))]),[1,2,3,4]);
const ablationTests = readCsv('report/resnet_ablation_statistical_tests.csv');
blocks.ABLATION_DELTAS = table(['Removed component','Δ Accuracy (pp)','Δ Macro F1 (pp)','Holm p: Accuracy','Holm p: Macro F1'],['no_augmentation','no_class_weights'].map(condition=>{const a=ablationTests.find(r=>r.condition===condition&&r.metric==='test_acc'),b=ablationTests.find(r=>r.condition===condition&&r.metric==='test_macro_f1');return [condition==='no_augmentation'?'Augmentation':'Class weights',f(a.mean_difference_pp),f(b.mean_difference_pp),pval(a.p_holm),pval(b.p_holm)];}),[1,2,3,4]);
let text = fs.readFileSync(path.join(__dirname,'readme_template.md'),'utf8');
text = text.replace(/\{\{([A-Z][A-Z0-9_]+)\}\}/g,(_,name)=>{
  if(!blocks[name])throw Error('Unknown report block: '+name);
  return '<!-- generated:'+name+' -->\n\n'+blocks[name]+'\n\n<!-- /generated:'+name+' -->';
});
if(/\{\{|TODO|ย้ายเนื้อหาเดิม|คงตารางเดิมไว้|ถ้าต้องการใส่ weight/.test(text))throw Error('Unresolved editorial instruction');
for(const key of Object.keys(before))if(hash(key)!==before[key])throw Error('Source mutated: '+key);
// Check local links, images and table widths before replacing the report.
const links = [...text.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)].map(m=>m[1]);
for(const href of links){if(/^(https?:|#)/.test(href))continue;const target=href.split('#')[0];if(target==='report/qa/presentation_validation.json')continue;if(!fs.existsSync(path.join(root,target)))throw Error('Broken README link: '+href);}
const lines=text.split(/\r?\n/);let columns=null,tables=0;
for(const line of lines){if(line.startsWith('|')){const count=line.split(/(?<!\\)\|/).length-2;if(columns===null){columns=count;tables++;}if(count!==columns)throw Error('Unequal Markdown table width: '+line);}else columns=null;}
fs.writeFileSync(path.join(root,'README.md'),text.trim()+'\n');
fs.writeFileSync(path.join(__dirname,'report_source.md'),'<!-- Generated mirror of ../README.md; edit report/readme_template.md and run node report/render_readme.js. -->\n'+text.trim().replaceAll('](report/','](')+'\n');
fs.mkdirSync(path.join(__dirname,'qa'),{recursive:true});
fs.writeFileSync(path.join(__dirname,'qa/presentation_validation.json'),JSON.stringify({
  checked_utc:new Date().toISOString(),source_sha256:before,readme_sha256:hash('README.md'),
  table_count:tables,local_and_external_links:links.length,broken_local_links:0,
  original_metric_units:'Saved CSV/JSON remain unchanged; only displayed metrics, differences and CIs are multiplied by 100. p/t/d_z are not rescaled.',
  model_selection:'ResNet50/Full post hoc; original MobileNetV3-Large/Partial statistical reference preserved.',
  execution_scope:'This renderer uses saved files only. The user separately authorized MobileNet Grad-CAM inference for eight fixed images; no new training or tuning was performed.',
  renderer_python_inference_training_performed:false
},null,2)+'\n');
console.log('Rendered README and report mirror; checked '+tables+' tables and '+links.length+' links.');
