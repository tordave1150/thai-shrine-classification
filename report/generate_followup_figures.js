// Re-layout saved CAM images and plot saved ablation metrics. No inference or training.
const fs = require('fs');
const path = require('path');
const { root, readCsv, hash, sharpRuntime, mean, sd } = require('./report_utils');
const theme = require('./figure_theme');
const assets = path.join(__dirname, 'report_assets/latest');
const sources = ['report/figure_theme.js', 'report/resnet_gradcam_cases.csv', 'report/resnet_followup_sources.json',
  'outputs_modes_v3_td/final/run_metrics.csv', 'outputs_modes_v3_td/ablation/resnet50_full_v1/run_metrics.csv',
  'outputs_modes_v3_td/ablation/resnet50_full_v1/condition_summary.csv'];
const originalHash = hash('report/report_assets/latest/gradcam_resnet50_full_seed42.png');
const conditions = ['baseline', 'no_augmentation', 'no_class_weights'];
const labels = ['Baseline', 'No augmentation', 'No class weights'];
const base = readCsv(sources[3]).filter(r => r.status === 'completed' && r.architecture === 'resnet50' && r.mode === 'full').map(r => ({ ...r, condition: 'baseline' }));
const runs = [...base, ...readCsv(sources[4]).filter(r => r.status === 'completed')];
const summary = readCsv(sources[5]);
const text = (x, y, value, size = 14, extra = '') => '<text x="' + x + '" y="' + y + '" text-anchor="middle" font-size="' + size + '" ' + extra + '>' + theme.esc(value) + '</text>';
const line = (x1, y1, x2, y2, color, width = 1) => '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + color + '" stroke-width="' + width + '"/>';
function diamond(x, y, r, opacity) { return '<path d="M ' + x + ' ' + (y-r) + ' l ' + r + ' ' + r + ' l ' + (-r) + ' ' + r + ' l ' + (-r) + ' ' + (-r) + ' Z" fill="' + theme.modeColors.full + '" opacity="' + opacity + '"/>'; }
let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 550" role="img" aria-label="ResNet50 Full ablation: five matched seeds"><rect width="1100" height="550" fill="white"/><style>' + theme.styles + '</style>';
svg += text(550, 34, 'ResNet50 / Full: ablation on test', 23, 'font-weight="700"');
svg += text(550, 65, '5 matched seeds · mean ± sample SD · diamonds = individual seeds', 14);
['test_acc', 'test_macro_f1'].forEach((metric, panel) => {
  const left = 75 + panel * 540, top = 128, width = 430, height = 280;
  const fy = value => top + height - (value - 80) / 20 * height;
  svg += text(left + width/2, 104, panel ? 'Macro F1 (%)' : 'Accuracy (%)', 17, 'font-weight="700"');
  svg += text(left-55, top+height/2, panel ? 'Macro F1 (%)' : 'Accuracy (%)', 14, 'transform="rotate(-90 '+(left-55)+' '+(top+height/2)+')"');
  for (const tick of [80, 85, 90, 95, 100]) {
    svg += line(left, fy(tick), left + width, fy(tick), theme.grid) + text(left - 24, fy(tick) + 5, tick, 13);
  }
  svg += line(left, top, left, top + height, theme.muted) + line(left, top + height, left + width, top + height, theme.muted);
  conditions.forEach((condition, i) => {
    const subset = runs.filter(r => r.condition === condition).sort((a, b) => Number(a.seed) - Number(b.seed));
    if (subset.length !== 5 || new Set(subset.map(r => r.seed)).size !== 5) throw Error('Incomplete matched seeds: ' + condition);
    const values = subset.map(r => Number(r[metric]) * 100), m = mean(values), s = sd(values);
    const saved = summary.find(r => r.condition === condition);
    if (Math.abs(m - Number(saved[metric + '_mean']) * 100) > 1e-9 || Math.abs(s - Number(saved[metric + '_sample_sd']) * 100) > 1e-9) throw Error('Ablation summary mismatch');
    if (Math.min(...values, m-s) < 80 || Math.max(...values, m+s) > 100) throw Error('Ablation axis would clip data');
    const cx = left + 65 + i * 150;
    svg += line(cx, fy(m-s), cx, fy(m+s), theme.modeColors.full, 2);
    for (const v of [m-s, m+s]) svg += line(cx-8, fy(v), cx+8, fy(v), theme.modeColors.full, 2);
    values.forEach((v, j) => { svg += diamond(cx + (j-2)*7, fy(v), 4, 0.4); });
    svg += diamond(cx, fy(m), 7, 1) + text(cx, 445, labels[i], 13);
    svg += text(cx, 471, m.toFixed(2) + ' ± ' + s.toFixed(2), 13);
  });
});
svg += text(550, 520, 'Display axes: 80–100 · all conditions use Full mode · values below plots: mean (%) ± SD (pp)', 12) + '</svg>\n';
const ablationFigure = 'report/report_assets/latest/resnet_ablation_summary.svg';
fs.writeFileSync(path.join(root, ablationFigure), svg);

async function camGrid() {
  const sharp = sharpRuntime();
  const cases = readCsv('report/resnet_gradcam_cases.csv');
  if (cases.length !== 8) throw Error('Expected eight saved CAM cases');
  const width = 2160, height = 2060, composites = [];
  const short = { san_phraphrom: 'พระพรหม', san_phraphum: 'พระภูมิ', san_tayay: 'ตายาย', tee_ju_eia: 'ตี่จู้เอี๊ยะ' };
  let captions = '<svg xmlns="http://www.w3.org/2000/svg" width="2160" height="2060"><style>text{font-family:' + theme.fontFamily + ';fill:' + theme.ink + '}</style>';
  captions += text(1080, 49, 'ResNet50 / Full · seed 42 · Grad-CAM at layer4[-1]', 38, 'font-weight="700"');
  captions += text(1080, 92, 'Selected examples · confidence (%) · 6 correct / 2 incorrect', 28);
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i], left = 35 + (i%2)*1080, top = 132 + Math.floor(i/2)*475;
    captions += '<rect x="' + (left-12) + '" y="' + (top-8) + '" width="1055" height="457" rx="9" fill="none" stroke="' + theme.grid + '" stroke-width="2"/>';
    captions += text(left+510, top+25, c.image_id.split('/')[1], 27, 'font-weight="700"');
    for (let j = 0; j < 3; j++) {
      const filename = ['original.png', 'heatmap.png', 'overlay.png'][j];
      const relative = 'outputs_modes_v3_td/' + c.sample_directory + '/' + filename;
      sources.push(relative);
      const input = path.join(root, relative);
      const resized = await sharp(input).resize(320, 320, {fit:'contain', background:'white'}).png().toBuffer();
      composites.push({input: resized, left: left+j*342, top: top+78});
      captions += text(left+j*342+160, top+61, ['Original', 'Grad-CAM', 'Overlay'][j], 25);
    }
    const correct = c.true_label === c.predicted_label;
    const label = 'จริง: ' + short[c.true_label] + ' → ทำนาย: ' + short[c.predicted_label] + ' | ' + (Number(c.confidence)*100).toFixed(2) + ' ' + (correct?'✓':'✗');
    captions += text(left+510, top+435, label, 29);
  }
  captions += '</svg>';
  composites.push({input: Buffer.from(captions), left:0, top:0});
  const figure = 'report/report_assets/latest/gradcam_resnet50_full_seed42_readable.png';
  await sharp({create:{width,height,channels:3,background:'white'}}).composite(composites).png().toFile(path.join(root, figure));
  if (hash('report/report_assets/latest/gradcam_resnet50_full_seed42.png') !== originalHash) throw Error('Original CAM grid changed');
  const inputs = Object.fromEntries([...new Set(sources)].map(p => [p, hash(p)]));
  fs.writeFileSync(path.join(__dirname, 'followup_figure_sources.json'), JSON.stringify({
    generated_utc: new Date().toISOString(), source_sha256: inputs,
    figures: [ablationFigure, figure], output_sha256: {[ablationFigure]:hash(ablationFigure), [figure]:hash(figure)},
    note: 'Presentation only: plotted saved run metrics and resized existing original/heatmap/overlay PNGs. No heatmap recomputation, inference, retuning, or training. Original eight-row grid preserved.'
  }, null, 2) + '\n');
  console.log('Generated ablation SVG and re-laid out eight saved CAM cases.');
}
camGrid().catch(error => { console.error(error); process.exitCode = 1; });
