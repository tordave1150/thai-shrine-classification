// Percentage labels for compatible inherited baseline/reference evidence.
// Reads saved predictions and images only; never runs an ImageNet or fine-tuned model.
const fs = require('fs');
const path = require('path');
const {root, readCsv, hash, sharpRuntime} = require('./report_utils');
const theme = require('./figure_theme');
const sharp = sharpRuntime();
const assets = 'report/report_assets/latest/';
const inputPaths = new Set(['report/figure_theme.js','outputs_modes_v3_td/baseline/pretrained_predictions.csv','outputs_modes_v3_td/eyeball/eyeball_cases.csv','outputs_modes_v3_td/splits.csv']);
const labels = {san_phraphrom:'พระพรหม',san_phraphum:'พระภูมิ',san_tayay:'ตายาย',tee_ju_eia:'ตี่จู้เอี๊ยะ'};
const text = (x,y,value,size=27) => '<text x="'+x+'" y="'+y+'" text-anchor="middle" font-size="'+size+'">'+theme.esc(value)+'</text>';
const pct = n => (Number(n)*100).toFixed(2);
function header(w,h,title,subtitle) { return '<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'"><style>text{font-family:'+theme.fontFamily+';fill:'+theme.ink+'}</style>'+text(w/2,48,title,38)+text(w/2,88,subtitle,27); }
async function thumb(relative,width,height) {
  inputPaths.add(relative);
  return sharp(path.join(root,relative)).resize(width,height,{fit:'contain',background:'white'}).png().toBuffer();
}
async function save(relative,width,height,svg,composites) {
  composites.push({input:Buffer.from(svg+'</svg>'),left:0,top:0});
  await sharp({create:{width,height,channels:3,background:'white'}}).composite(composites).png().toFile(path.join(root,relative));
  return relative;
}
async function run() {
  const outputs = [];
  const splits = new Map(readCsv('outputs_modes_v3_td/splits.csv').map(r=>[r.image_id,r]));
  const allBaseline = readCsv('outputs_modes_v3_td/baseline/pretrained_predictions.csv');
  if(allBaseline.length!==390)throw Error('Expected inherited ImageNet predictions for all 390 train images');
  const baseline = Object.keys(labels).flatMap(label => allBaseline.filter(r=>r.true_custom_class===label).slice(0,3));
  if(baseline.length!==12)throw Error('Missing three examples per class');
  const w=1800,h=1900,composites=[];
  let svg=header(w,h,'ResNet50 / ImageNet: original label space','Fixed train examples · ImageNet top-1 confidence (%)');
  for(let i=0;i<baseline.length;i++){
    const r=baseline[i],split=splits.get(r.image_id),left=20+(i%3)*600,top=120+Math.floor(i/3)*440;
    if(!split||split.split!=='train'||split.label!==r.true_custom_class)throw Error('Incompatible inherited baseline image');
    composites.push({input:await thumb('dataset_cleaned/'+split.path,550,300),left,top});
    svg+=text(left+275,top+330,r.image_id.split('/')[1],23)+text(left+275,top+368,'จริง: '+labels[r.true_custom_class],29);
    svg+=text(left+275,top+408,'ImageNet: '+r.imagenet_top1+' | '+pct(r.imagenet_confidence),28);
  }
  outputs.push(await save(assets+'imagenet_baseline_percent.png',w,h,svg,composites));

  const cases=readCsv('outputs_modes_v3_td/eyeball/eyeball_cases.csv');
  if(cases.length!==8)throw Error('Expected eight reference cases');
  const cw=2160,ch=2060,cam=[],ew=1800,eh=2100,eye=[];
  let cs=header(cw,ch,'MobileNetV3-Large / Partial · seed 42 · supplementary Grad-CAM','Saved reference examples · confidence (%)');
  let es=header(ew,eh,'MobileNetV3-Large / Partial · seed 42 · supplementary eyeball','ImageNet and fine-tuned predictions on the same image · confidence (%)');
  for(let i=0;i<cases.length;i++){
    const c=cases[i],dir='outputs_modes_v3_td/'+c.cam_files.replaceAll('\\','/'),left=35+(i%2)*1080,top=132+Math.floor(i/2)*475;
    cs+=text(left+510,top+25,c.image_id.split('/')[1],27);
    for(let j=0;j<3;j++){
      const file=['original.png','heatmap.png','overlay.png'][j];
      cam.push({input:await thumb(dir+'/'+file,320,320),left:left+j*342,top:top+78});
      cs+=text(left+j*342+160,top+61,['Original','Grad-CAM','Overlay'][j],25);
    }
    cs+=text(left+510,top+435,'จริง: '+labels[c.true_label]+' → ทำนาย: '+labels[c.predicted_label]+' | '+pct(c.confidence)+' '+(c.true_label===c.predicted_label?'✓':'✗'),29);
    const ex=30+(i%2)*900,ey=130+Math.floor(i/2)*480;
    eye.push({input:await thumb(dir+'/original.png',330,350),left:ex,top:ey+40});
    es+=text(ex+420,ey+12,c.image_id.split('/')[1],25);
    es+=text(ex+600,ey+135,'จริง: '+labels[c.true_label],30);
    es+=text(ex+600,ey+190,'ImageNet: '+c.imagenet_label,27)+text(ex+600,ey+233,pct(c.imagenet_confidence),29);
    es+=text(ex+600,ey+300,'Fine-tuned: '+labels[c.predicted_label],27)+text(ex+600,ey+343,pct(c.confidence)+' '+(c.true_label===c.predicted_label?'✓':'✗'),29);
  }
  outputs.push(await save(assets+'gradcam_reference_percent.png',cw,ch,cs,cam));
  outputs.push(await save(assets+'eyeball_reference_percent.png',ew,eh,es,eye));
  fs.writeFileSync(path.join(__dirname,'context_figure_sources.json'),JSON.stringify({generated_utc:new Date().toISOString(),
    source_sha256:Object.fromEntries([...inputPaths].map(p=>[p,hash(p)])),output_sha256:Object.fromEntries(outputs.map(p=>[p,hash(p)])),
    baseline_example_ids:baseline.map(r=>r.image_id),
    note:'Compatible inherited ImageNet baseline and latest reference diagnostics. First three logged train examples per class, fixed canonical class order. Re-layout and percentage conversion only. No inference/training. Original images and saved grids preserved.'},null,2)+'\n');
  console.log('Reformatted inherited baseline and reference examples with confidence (%).');
}
run().catch(error=>{console.error(error);process.exitCode=1;});
