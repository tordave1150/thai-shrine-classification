// Report-only figure from saved diagnostics. Run: node report/generate_selected_model_calibration.js
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const theme = require('./figure_theme');
const root = path.resolve(__dirname, '..');
const architecture = 'resnet50', mode = 'full', seed = '42';
const source = 'outputs_modes_v3_td/confidence/bin_metrics.csv';
const input = fs.readFileSync(path.join(root, source));
function csv(text) {
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  const fields = line => line.split(',').map(v => v.replace(/^"|"$/g, ''));
  const header = fields(lines.shift());
  return lines.map(line => Object.fromEntries(fields(line).map((v, i) => [header[i], v])));
}
const bins = csv(input.toString('utf8')).filter(r => r.architecture === architecture && r.mode === mode && r.seed === seed);
if (bins.length !== 10 || bins.reduce((n, r) => n + Number(r.count), 0) !== 85) throw Error('Incomplete saved calibration bins');
const x = 90, y = 90, width = 580, height = 370;
let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 570" role="img" aria-label="ResNet50 Full calibration on test, seed 42"><rect width="800" height="570" fill="white"/><style>text{font-family:Arial,"Noto Sans Thai","Leelawadee UI",Tahoma,sans-serif;fill:#1F2937;font-size:13px}.small{font-size:11px}</style>';
const text = (xx, yy, value, extra = '') => `<text x="${xx}" y="${yy}" text-anchor="middle" ${extra}>${value}</text>`;
svg += text(400, 34, 'ResNet50 / Full: test calibration, seed 42 (%)', 'style="font-size:23px;font-weight:700"');
for (const v of [0, 20, 40, 60, 80, 100]) {
  const py = y + height - v / 100 * height;
  svg += `<line x1="${x}" y1="${py}" x2="${x+width}" y2="${py}" stroke="#e7ebee"/>`;
  svg += text(x-19, py+4, v, 'class="small"') + text(x+v/100*width, y+height+18, v, 'class="small"');
}
svg += `<path d="M ${x} ${y} V ${y+height} H ${x+width}" fill="none" stroke="#87929b"/><line x1="${x}" y1="${y+height}" x2="${x+width}" y2="${y}" stroke="#c4cdd3" stroke-dasharray="7 5"/>`;
const seedEce = bins.reduce((sum, b) => sum + Number(b.count) / 85 * (Number(b.count) ? Math.abs(Number(b.mean_confidence)-Number(b.accuracy)) : 0), 0) * 100;
svg += text(265, 114, 'ECE (10 bins, this seed): ' + seedEce.toFixed(2), 'class="small"');
for (const b of bins) {
  if (!Number(b.count)) continue;
  const xx = x + Number(b.mean_confidence)*width, yy = y+height-Number(b.accuracy)*height;
  svg += `<circle cx="${xx}" cy="${yy}" r="${Math.max(5, Math.sqrt(Number(b.count))*2.6)}" fill="${theme.modeColors.full}" opacity=".65" stroke="${theme.ink}"/>` + text(xx, yy-13, b.count, 'class="small"');
}
svg += text(380, 499, 'Mean confidence (%)') + text(28, 275, 'Observed accuracy (%)', 'transform="rotate(-90 28 275)"') + text(400, 532, 'Dot area scales with bin count; label = images in bin (test n = 85)', 'class="small"') + '</svg>\n';
const figure = 'report/report_assets/latest/selected_resnet50_full_calibration.svg';
fs.writeFileSync(path.join(root, figure), svg);
const summaryPath = 'report/verified_pair_summary.csv';
const summaries = csv(fs.readFileSync(path.join(root, summaryPath), 'utf8'));
const metricLeaders = Object.fromEntries(['test_acc_mean','test_macro_precision_mean','test_macro_recall_mean','test_macro_f1_mean'].map(metric => {
  const best = [...summaries].sort((a,b) => Number(b[metric])-Number(a[metric]))[0];
  return [metric, {architecture:best.architecture, mode:best.mode, value:Number(best[metric])}];
}));
fs.writeFileSync(path.join(__dirname, 'selected_model_report_sources.json'), JSON.stringify({
  selected_report_model:{architecture,mode}, selection_context:'Post-hoc descriptive choice after examining multiple test metrics; requires confirmation on a new external holdout.',
  statistical_reference:{architecture:'mobilenet_v3_large',mode:'partial',original_selection_json_unchanged:true},
  metric_leaders:metricLeaders, calibration:{source,source_sha256:crypto.createHash('sha256').update(input).digest('hex'),seed:Number(seed),figure},
  pair_summary:{source:summaryPath,source_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,summaryPath))).digest('hex')},
  gradcam_resnet50_full: JSON.parse(fs.readFileSync(path.join(__dirname, 'resnet_followup_sources.json'), 'utf8')).gradcam
},null,2)+'\n');
console.log('Generated ResNet50/Full calibration from saved bins and recorded the report selection separately from the experiment lock.');
