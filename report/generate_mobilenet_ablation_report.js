// Report-only publication of the completed, separately locked MobileNet ablation.
// No model execution. Called by resnet_followup.py --architecture mobilenet_v3_large --stage report.
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),cp=require('child_process');
const {root,readCsv,hash,mean,sd,sharpRuntime}=require('./report_utils'),T=require('./figure_theme');
const sources={},read=p=>{sources[p]=hash(p);return fs.readFileSync(path.join(root,p),'utf8');},json=p=>JSON.parse(read(p).replace(/^\uFEFF/,'')),csv=p=>{read(p);return readCsv(p);};
const out='outputs_modes_v3_td/',source=json('report/mobilenet_ablation_sources.json'),study=out+source.study_directory+'/';
const lock=json(study+'ablation_lock.json'),registry=json(study+'run_registry.json'),original=json(out+'run_registry.json');
assert.equal(lock.architecture,'mobilenet_v3_large');assert.equal(lock.mode,'partial');assert.equal(lock.statistics.family_size,4);
const seeds=[42,123,456,789,1024],conditions=['baseline','no_augmentation','no_class_weights'];
const base=original.final_runs.filter(r=>r.architecture==='mobilenet_v3_large'&&r.mode==='partial'&&r.status==='completed');
const fresh=registry.attempts.filter(r=>r.status==='completed');assert.equal(fresh.length,10);
const runs=[...base.map(r=>({...r,condition:'baseline'})),...fresh];
const summary=csv('report/mobilenet_ablation_condition_summary.csv'),tests=csv('report/mobilenet_ablation_statistical_tests.csv');assert.equal(summary.length,3);assert.equal(tests.length,4);
for(const c of conditions){const rows=runs.filter(r=>r.condition===c).sort((a,b)=>a.seed-b.seed);assert.deepEqual(rows.map(r=>r.seed),seeds);for(const metric of ['test_acc','test_macro_precision','test_macro_recall','test_macro_f1']){const values=rows.map(r=>r[metric]),saved=summary.find(r=>r.condition===c);assert(Math.abs(mean(values)-saved[metric+'_mean'])<1e-10);assert(Math.abs(sd(values)-saved[metric+'_sample_sd'])<1e-10);}}
const signed=(n,d=2)=>(Number(n)>0?'+':'')+Number(n).toFixed(d),pct=n=>(100*Number(n)).toFixed(2),pm=(r,k)=>pct(r[k+'_mean'])+' ± '+pct(r[k+'_sample_sd']);
const pmUnits=(r,k)=>pct(r[k+'_mean'])+'% ± '+pct(r[k+'_sample_sd'])+' pp';
const pairSummary=csv('report/verified_pair_summary.csv'),mobile=pairSummary.find(r=>r.architecture==='mobilenet_v3_large'&&r.mode==='partial'),resnet=pairSummary.find(r=>r.architecture==='resnet50'&&r.mode==='full');
const mainTests=csv('report/verified_statistics.csv').filter(r=>r.reference==='mobilenet_v3_large::partial'&&r.competitor==='resnet50::full');assert.equal(mainTests.length,2);assert(mainTests.every(r=>Number(r.p_holm)>=.05));
const pc=csv('report/verified_per_class_summary.csv'),confidence=csv(out+'confidence/pair_summary.csv');
const mobileCfg=base.find(r=>r.seed===42),resRun=original.final_runs.find(r=>r.architecture==='resnet50'&&r.mode==='full'&&r.seed===42&&r.status==='completed');
const rel=p=>{const value=String(p).replaceAll('\\','/'),token='/outputs_modes_v3_td/';return value.includes(token)?'outputs_modes_v3_td/'+value.split(token)[1]:path.relative(root,p).replaceAll('\\','/');};
const mc=json(rel(path.join(mobileCfg.run_dir,'config.json'))),rc=json(rel(path.join(resRun.run_dir,'config.json')));
const mobileParams=mc.mode_stats.trainable_parameters,resParams=rc.mode_stats.trainable_parameters,reduction=100*(1-mobileParams/resParams);
const mobileTotal=mc.mode_stats.total_parameters,resTotal=rc.mode_stats.total_parameters,totalReduction=100*(1-mobileTotal/resTotal);
const mainEpochs=[...original.trials,...original.final_runs].filter(r=>r.status==='completed').reduce((n,r)=>n+r.epochs_completed,0);
const prior=json(out+'ablation/resnet50_full_v1/run_registry.json').attempts.filter(r=>r.status==='completed');
const actualEpochs=fresh.reduce((n,r)=>n+r.epochs_completed,0),allEpochs=mainEpochs+prior.reduce((n,r)=>n+r.epochs_completed,0)+actualEpochs;
const test=(c,m)=>tests.find(r=>r.condition===c&&r.metric===m),naF1=test('no_augmentation','test_macro_f1'),nwF1=test('no_class_weights','test_macro_f1');
const significant=tests.filter(r=>String(r.significant_holm).toLowerCase()==='true'),valid=tests.filter(r=>r.status==='ok'),minimum=valid.length?Math.min(...valid.map(r=>+r.p_holm)):null;
const width=1140,height=565,ink=T.modeColors.partial;
const txt=(x,y,v,size=15,extra='')=>`<text x="${x}" y="${y}" text-anchor="middle" font-size="${size}" ${extra}>${T.esc(v)}</text>`;
const line=(x,y,x2,y2,color=T.grid,w=1)=>`<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
const square=(x,y,r,o=1)=>`<rect x="${x-r}" y="${y-r}" width="${r*2}" height="${r*2}" fill="${ink}" opacity="${o}"/>`;
const measured=runs.flatMap(r=>[r.test_acc*100,r.test_macro_f1*100]),means=summary.flatMap(r=>['test_acc','test_macro_f1'].flatMap(k=>[(+r[k+'_mean']-+r[k+'_sample_sd'])*100,(+r[k+'_mean']+ +r[k+'_sample_sd'])*100]));
const lo=Math.min(80,Math.floor(Math.min(...measured,...means)/5)*5),hi=Math.max(100,Math.ceil(Math.max(...measured,...means)/5)*5);
let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"><rect width="${width}" height="${height}" fill="white"/><style>text{font-family:${T.fontFamily};fill:${T.ink}}</style>`;
svg+=txt(width/2,34,'MobileNetV3-Large / Partial: ablation on test',25,'font-weight="700"')+txt(width/2,65,'Five matched seeds · squares: individual seeds · large square and bars: mean ± sample SD',16);
for(const [panel,metric]of ['test_acc','test_macro_f1'].entries()){
 const left=85+panel*570,top=130,w=430,h=285,y=v=>top+h-(v-lo)/(hi-lo)*h;
 svg+=txt(left+w/2,107,panel?'Macro F1 (%)':'Accuracy (%)',20,'font-weight="700"');
 for(let tick=lo;tick<=hi;tick+=5)svg+=line(left,y(tick),left+w,y(tick))+txt(left-25,y(tick)+5,tick,14);
 svg+=line(left,top,left,top+h,T.muted)+line(left,top+h,left+w,top+h,T.muted);
 for(const [i,c]of conditions.entries()){const values=runs.filter(r=>r.condition===c).sort((a,b)=>a.seed-b.seed).map(r=>r[metric]*100),m=mean(values),s=sd(values),x=left+65+i*150;svg+=line(x,y(m-s),x,y(m+s),ink,2)+line(x-8,y(m-s),x+8,y(m-s),ink,2)+line(x-8,y(m+s),x+8,y(m+s),ink,2);values.forEach((v,j)=>svg+=square(x+(j-2)*7,y(v),4,.42));svg+=square(x,y(m),7)+txt(x,451,['Baseline','No augmentation','No class weights'][i],14)+txt(x,480,m.toFixed(2)+' ± '+s.toFixed(2),14);}
}
svg+=txt(width/2,526,'Same Partial mode and locked recipe in all conditions; each variant removes one factor.',16)+txt(width/2,552,`Baseline: five reused final runs · new variants: ten runs / ${actualEpochs} actual epochs`,15)+'</svg>\n';
const figure='report/report_assets/latest/mobilenet_ablation_comparison.svg';
function replaceParagraph(s,prefix,value){const aliases={'**ResNet50/Full ได้รับเลือกเป็นโมเดลสุดท้าย':'**MobileNetV3-Large/Partial ได้รับเลือกเป็นโมเดลสุดท้าย','การศึกษา ablation ใช้ตรวจสอบ':'การศึกษา ablation ของ MobileNetV3-Large/Partial','การทดลองเสร็จครบทั้ง 15 setups':'งานหลักเสร็จครบทั้ง 15 setups','จากผลประเมินหลายด้าน **ResNet50/Full เป็นโมเดลที่เลือก':'**MobileNetV3-Large/Partial เป็นโมเดลที่เลือกสำหรับงานนี้**','พระภูมิยังเป็นคลาสที่แยกได้ยากที่สุด และการเปรียบเทียบ Grad-CAM':'พระภูมิยังเป็นคลาสที่แยกได้ยากที่สุด Grad-CAM'};const p=s.split('\n\n'),ids=p.map((x,i)=>x.startsWith(prefix)||(aliases[prefix]&&x.startsWith(aliases[prefix]))?i:-1).filter(i=>i>=0);assert.equal(ids.length,1,prefix);p[ids[0]]=value;return p.join('\n\n');}
async function main(){
 fs.writeFileSync(path.join(root,figure),svg);await sharpRuntime()(Buffer.from(svg)).png().toFile(path.join(root,figure.replace('.svg','.png')));
 let doc=fs.readFileSync(path.join(root,'report/readme_template.md'),'utf8');
 doc=doc.replaceAll('6.10 Ablation study of ResNet50/Full','6.10 Ablation study of MobileNetV3-Large/Partial').replaceAll('#610-ablation-study-of-resnet50full','#610-ablation-study-of-mobilenetv3-largepartial');
 doc=replaceParagraph(doc,'**ResNet50/Full ได้รับเลือกเป็นโมเดลสุดท้าย','**MobileNetV3-Large/Partial ได้รับเลือกเป็นโมเดลสุดท้ายของการศึกษา** สอดคล้องกับโมเดลอ้างอิงที่เลือกจาก validation เดิม โดยมี test Accuracy '+pmUnits(mobile,'test_acc')+' และ Macro F1 '+pmUnits(mobile,'test_macro_f1')+' แม้ ResNet50/Full มีค่าเฉลี่ยสูงกว่าเล็กน้อย แต่ยังไม่พบความแตกต่างของ Accuracy และ Macro F1 อย่างมีนัยสำคัญหลังปรับ Holm (p = 1.000 ทั้งสอง metrics) งานนี้จึงให้น้ำหนักกับข้อจำกัดทรัพยากร: MobileNet/Partial มี trainable parameters '+mobileParams.toLocaleString('en-US')+' ตัว เทียบกับ '+resParams.toLocaleString('en-US')+' ตัวของ ResNet/Full น้อยกว่า '+reduction.toFixed(2)+'% และ Grad-CAM ที่ตรวจแสดงบริเวณตอบสนองใกล้ตัวศาลในหลายภาพ การไม่พบนัยสำคัญไม่ได้ยืนยันว่าโมเดลเท่ากัน จำนวน parameters ไม่ใช่ผลวัดความเร็วจริง และข้อสรุปจาก test split ที่เคยดูแล้วคงมีสถานะเชิงสำรวจ');
 doc=replaceParagraph(doc,'การศึกษา ablation ใช้ตรวจสอบ','การศึกษา ablation ของ MobileNetV3-Large/Partial เปรียบเทียบ baseline กับการนำ augmentation หรือ class weights ออกทีละอย่าง โดยใช้ matched seeds 5 ค่าและสูตรเดิม ไม่ tune หรือฝึก baseline ซ้ำ การนำ augmentation ออกทำให้ Macro F1 เปลี่ยน '+signed(naF1.mean_difference_pp)+' pp ส่วนการนำ class weights ออกเปลี่ยน '+signed(nwF1.mean_difference_pp)+' pp พบ '+significant.length+'/4 comparisons ที่มีนัยสำคัญหลัง Holm การศึกษาใหม่นี้ฝึก 10 runs / '+actualEpochs+' epochs และเก็บผล ResNet ablation เดิมไว้เป็นหลักฐานย้อนหลัง รวมงานที่ฝึกจริงทั้งหมด 170 runs / '+allEpochs.toLocaleString('en-US')+' epochs');
 doc=doc.replace('เหตุผลที่เลือก ResNet50/Full จากผลการทดลองอยู่ในหัวข้อ 6.9','เหตุผลที่เลือก MobileNetV3-Large/Partial โดยพิจารณาทั้งผลทดสอบและข้อจำกัดทรัพยากรอยู่ในหัวข้อ 6.9');
 doc=replaceParagraph(doc,'การทดลองเสร็จครบทั้ง 15 setups','งานหลักเสร็จครบทั้ง 15 setups: setup ละ 5 tuning trials และ 5 final seeds รวม 75 tuning + 75 final runs โดยประเมินบน test 85 ภาพชุดเดียวกันเฉพาะ final runs หลังล็อกสูตรแล้ว หัวข้อ 6.10 แสดง MobileNet/Partial ablation ที่ฝึกเพิ่ม 10 runs / '+actualEpochs+' epochs โดย reuse baseline เดิม 5 seeds ส่วน ResNet ablation เดิม 10 runs / 145 epochs เก็บไว้เป็นหลักฐานย้อนหลัง รวมงานที่ฝึกจริง 170 runs / '+allEpochs.toLocaleString('en-US')+' epochs โดยไม่บวก baseline ซ้ำ');
 doc=doc.replace('ผลจึงสนับสนุนการเลือกจากหลายด้านร่วมกัน ตารางแสดง mean (%) ± sample SD (pp) จาก 5 seeds เรียงตาม Macro F1','อันดับนี้เป็นผลเชิงพรรณนาบน test ส่วนการเลือก MobileNet/Partial โดยพิจารณาทรัพยากรร่วมด้วยอธิบายในหัวข้อ 6.9 ตารางแสดง mean (%) ± sample SD (pp) จาก 5 seeds เรียงตาม Macro F1');
 doc=doc.replace('เปรียบเทียบ ResNet50/Full (โมเดลที่เลือก) กับ MobileNetV3-Large/Partial (โมเดลอ้างอิง)','เปรียบเทียบ MobileNetV3-Large/Partial (โมเดลที่เลือกและ validation reference) กับ ResNet50/Full (โมเดลเปรียบเทียบ)');
 doc=doc.replace('![Confusion matrix ของ ResNet50/Full](report/report_assets/latest/confusion_resnet50_full.svg)','![Confusion matrix ของ MobileNetV3-Large/Partial](report/report_assets/latest/confusion_mobilenet_v3_large_partial.svg)');
 doc=doc.replace('*Figure 16 Confusion matrix ของ ResNet50/Full (ค่าเฉลี่ย % และ SD เป็น pp ข้าม 5 seeds) พระภูมิถูกทายถูกเฉลี่ย 77.8% ± 3.9 pp ซึ่งต่ำสุดในสี่คลาส*','*Figure 16 Confusion matrix ของ MobileNetV3-Large/Partial (ค่าเฉลี่ย % และ SD เป็น pp ข้าม 5 seeds) พระภูมิถูกทายถูกเฉลี่ย 72.2% ± 10.4 pp ซึ่งต่ำสุดในสี่คลาส*');
 const mobileRecall=pc.find(r=>r.architecture==='mobilenet_v3_large'&&r.mode==='partial'&&r.class==='san_phraphum'),resRecall=pc.find(r=>r.architecture==='resnet50'&&r.mode==='full'&&r.class==='san_phraphum');
 const ece=(a,m)=>{const r=confidence.find(r=>r.architecture===a&&r.mode===m);assert(Number.isFinite(+r.test_ece_10bin_mean)&&Number.isFinite(+r.test_ece_10bin_sample_sd));return pct(r.test_ece_10bin_mean)+' ± '+pct(r.test_ece_10bin_sample_sd);};
 const selection=`### 6.9 Final model selection

**เลือก MobileNetV3-Large/Partial เป็นโมเดลสุดท้าย** เพื่อให้ตรงกับโจทย์ด้านทรัพยากร โดยยังใช้สูตร baseline ที่มี augmentation และ class weights และสอดคล้องกับ validation-selected reference ใน selection.json เดิม ไม่เปลี่ยนสูตร checkpoint หรือแผนสถิติหลังเห็น test

| Metric | MobileNetV3-Large/Partial (selected) | ResNet50/Full (comparison) |
| --- | ---: | ---: |
${['test_acc','test_macro_precision','test_macro_recall','test_macro_f1'].map((k,i)=>'| '+['Accuracy (%) ↑','Macro precision (%) ↑','Macro recall (%) ↑','Macro F1 (%) ↑'][i]+' | '+pm(mobile,k)+' | '+pm(resnet,k)+' |').join('\n')}
| Phra Phum recall (%) ↑ | ${pm(mobileRecall,'recall')} | ${pm(resRecall,'recall')} |
| ECE (10 bins, %) ↓ | ${ece('mobilenet_v3_large','partial')} | ${ece('resnet50','full')} |
| Total parameters (จำนวนตัว) | ${mobileTotal.toLocaleString('en-US')} | ${resTotal.toLocaleString('en-US')} |
| Trainable parameters (จำนวนตัว) | ${mobileParams.toLocaleString('en-US')} | ${resParams.toLocaleString('en-US')} |

ResNet มีค่าเฉลี่ย Accuracy สูงกว่า 0.47 pp และ Macro F1 สูงกว่า 0.88 pp แต่ paired tests จาก 5 matched seeds ยังไม่พบความแตกต่างหลัง Holm (p = 1.000 ทั้งสอง metrics) การไม่ปฏิเสธสมมติฐานนี้ไม่ใช่หลักฐานว่าคะแนนเท่ากัน จึงใช้เป็นบริบทประกอบการตัดสินใจตามโจทย์ทรัพยากร ไม่ใช่ข้อพิสูจน์ว่า MobileNet ไม่ด้อยกว่า

MobileNet/Partial มี **trainable parameters น้อยกว่า ${reduction.toFixed(2)}%** และ total parameters น้อยกว่า ${totalReduction.toFixed(2)}% ภาระการปรับน้ำหนักและการเก็บ gradient/optimizer state จึงมีขนาดเล็กกว่าตามจำนวน parameters แต่ยังไม่มี benchmark ที่ควบคุมเงื่อนไขสำหรับเวลา หน่วยความจำ พลังงาน หรือ inference latency จึงไม่รายงานว่าเร็วขึ้นเป็นกี่เท่า

Grad-CAM ใน Figure 17 แสดงว่า MobileNet ตอบสนองบริเวณเรือน ฐาน หรือยอดศาลได้ในหลายตัวอย่าง เช่น san_phraphum_0043 และ san_tayay_0093 ที่ MobileNet ทายถูกขณะที่ ResNet ทายผิด อย่างไรก็ตามยังมีภาพที่ MobileNet ผิดแต่ ResNet ถูก และภาพที่ทั้งคู่ผิดด้วย confidence สูง หลักฐานนี้เป็นการตรวจเชิงคุณภาพ ไม่ใช่คะแนนการระบุตำแหน่งหรือหลักฐานเชิงสาเหตุ

การเลือกสุดท้ายจึงให้น้ำหนักกับขนาดโมเดล/จำนวน weights ที่ต้องปรับร่วมกับผลจำแนกและ CAM ที่ตรวจ ไม่ได้อาศัย p-value เพียงอย่างเดียว โมเดลที่เลือกตรงกับ validation reference เดิม แต่การอธิบายประกอบจาก test และ ablation บน split ที่เคยดูแล้วคงเป็น exploratory และไม่เปลี่ยน selection.json หรือ Holm families 28/30 เดิม

ไฟล์ประกอบ: [หลักฐานการเลือกโมเดล](report/final_model_selection.json)

`;
 doc=doc.replace(/### 6\.9 Final model selection[\s\S]*?(?=### 6\.10 Ablation study)/,selection);
 const statsText=significant.length?'พบ '+significant.length+' comparisons ที่มีนัยสำคัญหลัง Holm; ดูทิศทางและขนาดผลต่างราย metric ในตาราง':'ยังไม่พบ comparisons ที่มีนัยสำคัญหลัง Holm ทั้ง 4 รายการ';
 const ablation=`### 6.10 Ablation study of MobileNetV3-Large/Partial

<!-- mobilenet-followup:ablation:start -->

ใช้ **MobileNetV3-Large/Partial** และ recipe ของคู่นี้ที่เลือกจาก validation เดิม เปรียบเทียบ baseline ซึ่งมี augmentation และ class weights กับสอง variants: ไม่ใช้ augmentation แต่คง class weights และไม่ใช้ class weights แต่คง augmentation เปลี่ยนทีละปัจจัย โดยคง architecture, Partial modules, optimizer, learning rate, weight decay, batch size, split, seeds และเกณฑ์เลือก checkpoint เดิม

ล็อก variants และ **Holm family แยก 4 comparisons** ก่อนฝึกใหม่จาก ImageNet: variant ละ 5 seeds รวม **10 runs / ${actualEpochs} actual epochs** (งบสูงสุด 300 epochs) แล้วจับคู่กับ baseline final เดิม 5 seeds ไม่มีการ tune ใหม่หรือฝึก baseline ซ้ำ

{{ABLATION_TABLE}}

![Ablation ของ MobileNetV3-Large/Partial](report/report_assets/latest/mobilenet_ablation_comparison.svg)

*Figure 19. Test Accuracy และ Macro F1 ของ MobileNetV3-Large/Partial: สี่เหลี่ยมเล็กแทนแต่ละ seed สี่เหลี่ยมใหญ่และเส้นแทน mean ± sample SD จาก 5 matched seeds ทุกเงื่อนไขใช้ Partial mode เดียวกัน*

<!-- mobilenet-followup:ablation:end -->

{{ABLATION_DELTAS}}

Δ คือคะแนน variant ลบ baseline โดยจับคู่ seed เดียวกัน: ค่าลบหมายถึงคะแนนลดลงเมื่อนำองค์ประกอบออก การไม่ใช้ augmentation ให้ Δ Accuracy ${signed(test('no_augmentation','test_acc').mean_difference_pp)} pp และ Δ Macro F1 ${signed(naF1.mean_difference_pp)} pp ส่วนการไม่ใช้ class weights ให้ ${signed(test('no_class_weights','test_acc').mean_difference_pp)} pp และ ${signed(nwF1.mean_difference_pp)} pp ตามลำดับ

ทดสอบแบบ paired สองด้านสำหรับ Accuracy และ Macro F1 ของแต่ละ variant เทียบ baseline รวม 4 comparisons แล้วปรับ Holm แยกจากงานหลัก ${statsText}${minimum===null?'':'; Holm-adjusted p ต่ำสุด '+minimum.toFixed(4)} ผลสะท้อนความแปรผันข้าม training seeds บน test split เดียว ไม่ใช่การยืนยันกับข้อมูลภายนอก และการไม่พบนัยสำคัญไม่ใช่หลักฐานว่าเงื่อนไขต่าง ๆ เท่ากัน

คง baseline ที่ล็อกจาก validation เดิมเป็นสูตรของโมเดลสุดท้าย ไม่เลือก variant หรือ retune จากผล ablation บน test; loss ของเงื่อนไขใช้และไม่ใช้ class weights อยู่คนละสเกลวัตถุประสงค์ การศึกษานี้เป็น exploratory follow-up บน split ที่เคยตรวจดูแล้ว

เก็บ console log, configuration, history ราย epoch, checkpoint, test predictions และภาพ curves/confusion ของแต่ละ run ไว้ใน [โฟลเดอร์การศึกษา](${study}) ผล ResNet ablation เดิมคงเก็บเป็น [หลักฐานย้อนหลัง](report/resnet_ablation_condition_summary.csv) และไม่ได้นำมาแทนผล MobileNet รอบนี้

ไฟล์ประกอบ: [matched deltas](report/mobilenet_ablation_matched_deltas.csv) · [statistical tests](report/mobilenet_ablation_statistical_tests.csv) · [per-class summary](report/mobilenet_ablation_per_class_summary.csv) · [แหล่งผลการทดลอง](report/mobilenet_ablation_sources.json) · [รายการ logs และภาพราย run](report/mobilenet_ablation_artifacts.json) · [การตรวจคำนวณจากผลทำนาย](report/mobilenet_ablation_verification.json)

<p align="right"><a href="#top">🔝 กลับสู่ด้านบน</a></p>

---

`;
 doc=doc.replace(/### 6\.10 Ablation study[\s\S]*?(?=## 7\. Discussion and conclusions)/,ablation);
 doc=replaceParagraph(doc,'จากผลประเมินหลายด้าน **ResNet50/Full เป็นโมเดลที่เลือก','**MobileNetV3-Large/Partial เป็นโมเดลที่เลือกสำหรับงานนี้** โดยคง validation-selected reference เดิมและให้ความสำคัญกับข้อจำกัดทรัพยากร แม้ค่าเฉลี่ย test ของ ResNet50/Full สูงกว่า แต่ Accuracy และ Macro F1 ยังไม่ต่างอย่างมีนัยสำคัญหลัง Holm ขณะที่ MobileNet/Partial มี trainable parameters น้อยกว่า '+reduction.toFixed(2)+'% และ Grad-CAM ที่ตรวจแสดงบริเวณตอบสนองบนตัวศาลในหลายตัวอย่าง จึงตอบโจทย์การใช้งานที่ให้น้ำหนักกับขนาดโมเดล โดยไม่อ้างว่าความไม่พบนัยสำคัญพิสูจน์ความเท่ากันหรือยืนยันความเร็วจริง');
 doc=replaceParagraph(doc,'พระภูมิยังเป็นคลาสที่แยกได้ยากที่สุด และการเปรียบเทียบ Grad-CAM','พระภูมิยังเป็นคลาสที่แยกได้ยากที่สุด Grad-CAM ช่วยให้ตรวจตัวอย่างของโมเดลที่เลือกได้ แต่การตอบสนองตรงตัวศาลยังไม่รับประกันการทายถูก ผล MobileNet ablation ใช้อธิบายผลต่างเมื่อเอา augmentation หรือ class weights ออก ไม่ใช้ปรับสูตรจาก test จึงคง baseline เดิมที่ล็อกจาก validation และอ่านผลเชิงคุณภาพร่วมกับคะแนนรายคลาส calibration และความแปรผันระหว่าง seeds');
 doc=doc.replace('และเลือก ResNet หลังพิจารณา test ผลจึงเป็นเชิงสำรวจ','แม้โมเดลสุดท้ายตรงกับ MobileNet ที่เลือกจาก validation เดิม เหตุผลประกอบจาก test และ follow-up ยังคงเป็นเชิงสำรวจ');
 assert(!/\*\*เลือก ResNet50\/Full เป็นโมเดลสุดท้าย|Ablation study of ResNet50\/Full|Appendix|ภาคผนวก/.test(doc));
 fs.writeFileSync(path.join(root,'report/readme_template.md'),doc);
 const decision={prepared_utc:new Date().toISOString(),selected_report_model:{architecture:'mobilenet_v3_large',mode:'partial'},original_validation_reference_preserved:true,original_selection_sha256:hash(out+'selection.json'),main_holm_families_unchanged:true,main_comparisons:mainTests,trainable_parameters:{mobilenet_partial:mobileParams,resnet_full:resParams,reduction_pct:reduction},total_parameters:{mobilenet:mobileTotal,resnet:resTotal,reduction_pct:totalReduction},resource_evidence_scope:'Parameter counts only; no controlled latency/memory/energy benchmark.',gradcam_evidence:'Qualitative localization in the inspected seed-42 examples; not causal or segmentation proof.',selection_reason:'Resource-aware choice aligned with the original validation reference; no detected Accuracy/F1 difference after Holm is context, not equivalence.',mobile_ablation:{study_directory:source.study_directory,new_runs:10,actual_epochs:actualEpochs,baseline_reused:5,significant_comparisons:significant.length},all_completed_training_runs:170,all_completed_epochs:allEpochs,exploratory_test_status:true,source_sha256:sources,figure_sha256:{[figure]:hash(figure),[figure.replace('.svg','.png')]:hash(figure.replace('.svg','.png'))}};
 fs.writeFileSync(path.join(root,'report/final_model_selection.json'),JSON.stringify(decision,null,2)+'\n');
 // Verification file is prepared by the independent verifier before rendering.
 cp.execFileSync(process.execPath,[path.join(__dirname,'verify_mobilenet_ablation.js')],{cwd:root,stdio:'inherit'});
 cp.execFileSync(process.execPath,[path.join(__dirname,'render_readme.js')],{cwd:root,stdio:'inherit'});
 console.log(JSON.stringify({selected_model:'MobileNetV3-Large/Partial',completed_new_runs:10,actual_epochs:actualEpochs,trainable_reduction_pct:reduction,significant_ablation_comparisons:significant.length}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
