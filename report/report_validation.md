# Report validation — 20 October 2026

## Academic and visual revision — 22 October 2026

- README headings, subheadings, disclosure summaries and table headers now use English sentence case. Thai narrative is retained without personal pronouns. The output-directory/date banner was removed from the report header.
- Added the agreed explanations of group-based splitting, train-derived class weights, EXIF/alpha handling, approximate RGB-weighted brightness, ImageNet mean/SD, augmentation, DataLoader, classifier scoring, fine-tuning versus train/eval mode, checkpoint selection and ten equal-width ECE bins.
- Kept the post-hoc ResNet50/Full choice and MobileNetV3-Large/Partial statistical reference separate. The test split's prior exposure, calibration trade-off and nonsignificant ablation comparisons remain explicit.
- Figures 3 and 5 use common-bin overlapping step histograms with within-class percentages (not density). All eight figure/class groups retain their original image counts and sum to 100%. Figure 5 includes class means and the overall mean. Colour modes use direct count/share labels and patterns, with an expanded non-RGB panel.
- Seven tuning panels use 75 real selected-epoch observations, including both training and validation accuracy. A separate plot shows all five configurations inside each of the 15 predeclared setups; only those within-setup winners are ringed.
- The selected-hyperparameter table lists learning rate, weight decay, optimizer and batch size for all 15 setups. The checkpoint table separates seed-specific selection, with all 75 identities, paths and history hashes in `final_checkpoint_summary.csv`.
- Five architecture figures cover all 15 final setups: loss, accuracy, validation Macro F1 and available-run counts. Aggregations use actual observations, sample SD with ddof=1, no imputation, and no SD at n=1. Per-run selected epochs are marked separately from the mean curve. Complete SD bands may mathematically extend outside the metric's possible domain; this is disclosed and the saved run values are unaltered.
- Independent report-only checks recomputed all 1,800 epoch/metric rows with zero mean/SD discrepancy, verified 210 n=1 rows have no sample SD, and confirmed all 860 measured CSV/JSON hashes are unchanged. No Python, model inference, training, smoke or model-test suite was run.
- All 35 SVGs parsed as XML. Offline Markdown rendering through a headless browser found 27 images with no broken image or horizontal overflow. Abstract, configuration table and learning-curve screenshots were reviewed, along with colour and grayscale figure proofs. Grayscale checks support readability; they are not a claim of clinical certification for every colour-vision condition.
- Current machine-readable checks: [report_revision_validation.json](qa/report_revision_validation.json), [readme_render_validation.json](qa/readme_render_validation.json) and [enhanced_figure_sources.json](enhanced_figure_sources.json). Earlier checks below describe the preceding report revisions.

## การปรับรูปแบบตามข้อเสนอที่ทบทวนแล้ว

- ใช้ `readme_template.md` และ `render_readme.js` เพื่อจัดภาษา หน่วย ตาราง และภาคผนวก โดยอ่านผลที่บันทึกไว้เท่านั้น อัปเดต `report_source.md` เป็นสำเนาของ README ฉบับเดียวกัน
- ตาราง Accuracy/Precision/Recall/F1/ECE/Confidence แสดง mean (%) และ SD (pp) ผลต่างและ CI แสดง pp ส่วน p, t และ d_z คงหน่วยเดิม ไฟล์ตัวเลขต้นทางไม่ถูกแปลงหน่วย
- Frozen ใช้น้ำเงิน/วงกลม, Partial ใช้ส้มแดง/สี่เหลี่ยม, Full ใช้เขียว/ข้าวหลามตัด กราฟสำรวจข้อมูลใช้ชุดสีคลาสแยกกัน Confusion matrices ใช้ heat scale เดียวกันทุกคู่ พร้อมตัวเลขสีขาวบนช่องสีเข้ม
- คง box plots ห้าโมเดลแยกภาพเพื่อให้อ่านได้ ช่วงแกนเหมือนกันทุกภาพอยู่แล้ว: F1 75–100 และ best epoch 0–30
- Ablation SVG วาดใหม่จาก saved run metrics และตรวจ mean/sample SD ตรงกับ condition summary แสดงคะแนนทั้งห้า seeds และช่วงแกน 80–100 อย่างชัดเจน
- จัด Grad-CAM เป็นสองคอลัมน์ × สี่แถวจาก original/heatmap/overlay ที่มีอยู่ ไม่คำนวณ heatmap ใหม่ พร้อมป้ายคลาสจริง คลาสที่ทำนาย confidence (%) และเครื่องหมายถูก/ผิด เก็บกริดต้นฉบับไว้
- จัดภาพ ImageNet baseline และ MobileNet reference ใหม่โดยใช้ predictions และภาพเดิม เพื่อให้ confidence เป็นเปอร์เซ็นต์ มีที่มาใน `context_figure_sources.json` ส่วนภาพ padding/augmentation ยังคงเป็นภาพสาธิตนโยบายจาก train ไม่ใช่ minibatch ใน training log
- แก้ถ้อยคำ ablation ให้ระบุเพียงว่าค่าเฉลี่ย baseline สูงกว่า แต่ยังไม่พบนัยสำคัญหลัง Holm ไม่อ้างความเท่ากันหรือความไม่ด้อยกว่า และคงชื่อคอลัมน์ loss แยกตัวช่วงท้ายแทนการวินิจฉัย overfitting โดยตรง
- คง ResNet50/Full เป็นโมเดลที่เลือกภายหลังดู test พร้อมข้อแลกเปลี่ยนด้าน calibration และข้อจำกัดเรื่อง external holdout ส่วน MobileNetV3-Large/Partial ยังคงเป็น reference ใน lock และครอบครัวสถิติเดิม
- ตรวจโครงสร้าง 24 ตาราง ลิงก์ 83 รายการ และ XML ของ 29 SVG พร้อมดูภาพที่เรนเดอร์จริง ดู `qa/presentation_validation.json` และ source maps ของแต่ละตัวสร้างรูป
- เทียบ SHA256 ของหลักฐานเดิม 86 ไฟล์กับ provenance ก่อนแก้รายงาน ไม่พบไฟล์ที่เปลี่ยน ดู `qa/evidence_preservation.json` ไม่มีการรัน Python, inference, training หรือชุดทดสอบโมเดล

## สิ่งที่ตรวจแล้ว

- ใช้resultsเฉพาะ `outputs_modes_v3_td` และตรวจว่ามีhistoricaloutputอื่นในworkspaceจริง โดยคัดออกอย่างชัดเจน ไม่มีการลบ/เขียนทับrawexperimentoutputs
- ครบ75uniquefinalrunidentitiesและ15pairsคู่ละ5matchedseeds; canonicaltestorderตรงทุกrun (85ภาพ/run)
- คำนวณAccuracy,macroP/R/F1และper-classP/R/F1/supportใหม่จากpredictionCSVsรวม6,375rows; zero_division=0
- คำนวณmean/sampleSD(ddof1)ใหม่ทั้ง15pairsและ60classrows ไม่ใช้screenshotหรือmaximumseedเป็นหลักฐานแทน
- ตรวจgroupsplitoverlapจากsplits.csv=0; ไม่ถือว่าครอบคลุมsource/sessionleakageที่ไม่ทราบ
- Pairedseedjoinsตรงกันและไม่พบduplicatehypotheses; independentlyrecomputedt/rawpและHolm28+30comparisons
- เพิ่มCohen'sd_zและ95%CIของpairedmeanโดยไม่เลือกcomparisonfamiliesใหม่จากผลtest
- README ฉบับล่าสุดแสดง box plots ของ selected checkpoints แทน mean ± SD temporal curves ที่อ่านยาก; ยังเก็บลิงก์ภาพ temporal curves และ source map เดิมไว้
- READMEมีหัวข้อหลัก7ส่วนตามลำดับ,unitsของAccuracyเป็น%และF1/P/Rเป็น0–1,ระบุmode/seed/aggregationไว้ใต้รูป และไม่สรุปว่าnon-significant=equal
- README และ report_source ใช้ relative paths สำหรับ GitHub; ตาราง generated blocks เดิมผ่านการตรวจจำนวนคอลัมน์ และตรวจลิงก์ local ของฉบับล่าสุดอีกครั้งหลังเพิ่ม SVG

## ผลตรวจตัวเลข

| Check | Maximum absolute discrepancy |
|---|---:|
| Prediction-derivedmetrics vs per-runJSON | 1.1102230246251565e-16 |
| Mean/SD vs existingpairsummary | 3.3306690738754696e-16 |
| Perclasssummaries/support | 2.220446049250313e-16 |
| Pairedt/rawp/Holm | 3.375077994860476e-14 |

ความต่างระดับfloating-pointเท่านั้น ไม่พบmetricconflictที่ต้องเปลี่ยนข้อสรุป 58statisticalrowsคำนวณได้และ5rowsผ่านHolm (primary2/28,secondary3/30) การแก้rounding/precisionไม่ได้เปลี่ยนfamilyหรือthreshold

วิธีverificationใช้PowerShellอ่านsavedCSVs/JSONs ไม่มีPython/tests/smoke/trainingใหม่ สำหรับn=5(df4)คำนวณtwo-sidedStudentttailอย่างอิสระด้วย `u=|t|/sqrt(t²+4); p=.5(1-u)²(u+2)` แล้วปรับHolmภายในfamily;CIใช้tcritical2.776445105 เป็น**unadjusted95%CI** ไม่ใช่simultaneousCI

ผลตรวจmachine-readable: [verification.json](verification.json) และ [verified_statistics.csv](verified_statistics.csv)

## การปรับ README และกราฟล่าสุด

- สร้างกราฟล่าสุดด้วย [generate_latest_figures.js](generate_latest_figures.js) จาก 75 tuning trials, 75 final runs, 6,375 test predictions และ audit 560 ภาพ; ไม่รัน Python, tests หรือ training
- Box plots แสดง validation Macro F1 และ best epoch ของแต่ละ final seed ส่วน mean ± SD curves เป็น supplementary link; การหยุดฝึกต่าง epoch ทำให้จำนวน available runs ในภาพเส้นช่วงท้ายต่างกัน
- Confusion matrices ของทุก 15 คู่เป็นเปอร์เซ็นต์ต่อ true-class row แล้วคำนวณ mean/sample SD ข้าม 5 seeds มีสเกลสี 0–100 เดียวกัน และ [ค่ารายช่อง](confusion_mean_sd.csv)
- สี/สัญลักษณ์ในกราฟ mode คงที่: Frozen น้ำเงิน/วงกลม, Partial ส้มแดง/สี่เหลี่ยม, Full เขียว/ข้าวหลามตัด ส่วนสีของคลาสใช้เฉพาะ EDA
- ข้อมูล [brightness](derived_brightness.csv) คำนวณจากภาพต้นฉบับ 560 ภาพด้วย Sharp, EXIF-aware decode และ Rec.709 luminance; ไม่เอาตัวเลขจากภาพตัวอย่างของผู้ใช้
- รายงานเลือก ResNet50/Full จากหลาย test metrics แบบ post-hoc โดยแยกจาก reference ของสถิติเดิม ไม่แก้ lock/families หรืออ้างว่าผล Holm ยืนยันความเหนือกว่า Confusion matrix, Grad-CAM และ calibration หลักเป็นของ ResNet; CAM/eyeball ของ MobileNet อยู่ในส่วนประกอบ
- ผู้ใช้รัน `resnet_followup.py` จนเสร็จ: มี ResNet CAM 8 cases และ ablation 10 runs/145 epochs ที่ล็อกสองเงื่อนไขและ Holm family 4 แยกไว้ Agent ตรวจ saved artifacts และคำนวณซ้ำด้วย Node เท่านั้น ไม่มีการรัน Python/inference/training เพิ่มระหว่างแก้รายงาน
- [ผลตรวจ follow-up](resnet_followup_verification.json): baseline 5 + ablation 10 = 15 runs/1,275 prediction rows; metrics/summary/deltas ตรงภายใน 1.2e-14 และ paired t/raw p/Holm ภายใน 3.4e-15; SHA256 ของรูป/CSV สำเนาตรงกับต้นทาง ไม่มี ablation comparison ที่ significant หลัง Holm
- เพิ่มภาพ padding ขนาดใหญ่จากภาพ train จริงหนึ่งภาพ และกริด augmentation 4 คลาส × 5 ภาพจาก canonical train split; เก็บ IDs, SHA256, seeds และพารามิเตอร์ใน [preprocessing_preview_sources.json](preprocessing_preview_sources.json) โดยระบุใน README ว่าเป็นภาพสาธิตที่สร้างใหม่ ไม่ใช่ minibatch ใน log
- แก้ตารางสัดส่วนข้อมูลที่มี `%` แยกเป็นอีกคอลัมน์ ตารางสถิติที่ comparison แตกเป็นหลาย cells และตารางรายคลาสที่ค่ารวมอยู่ cell เดียว แก้การจัดกลุ่ม expression ในตัวสร้างตารางและเพิ่มการตรวจจำนวน cells ก่อนเขียนไฟล์
- ย่อตารางสูตรชนะให้อ้าง trial ในตาราง search configurations และปรับภาษาเนื้อหารายงาน โดยคงค่าผลทดลองและข้อสรุปทางสถิติเดิม
- การปรับรอบนี้ใช้ PowerShell อ่านผลเดิมและ Node.js สร้างกราฟ/ตรวจลิงก์ ไม่ได้รัน Python, training หรือชุดทดสอบโค้ด

## ข้อจำกัด/สิ่งที่ยังไม่ยืนยัน

1. **[TODO — source evidence not found]** PDF guideline/exampleต้นฉบับ ไม่มีในโครงการที่ค้น จึงใช้7sectionorderจากinstructionMDและคำขอล่าสุดที่ให้MarkdownแทนPDF ไม่อ้างว่าตรวจPDFแล้ว
2. **[TODO — source evidence not found]** Collection/scraperprotocollogที่บอกquery/date/manual-vs-scrapedแต่ละภาพและlicenseครบทุกภาพ ไม่มีหลักฐานพอ รายงานใช้URLmetadataที่มีจริงและระบุunknown112ภาพ
3. Brightness values ไม่อยู่ใน audit เดิม จึงคำนวณใหม่จากภาพต้นฉบับด้วยวิธีที่บันทึกใน `generate_latest_figures.js` และเก็บค่ารายภาพไว้ใน `derived_brightness.csv`; ไม่ตีความเป็นผลของการฝึก
4. มี ResNet50/Full Grad-CAM seed 42 ที่ผู้ใช้สร้างเสร็จแล้ว และ MobileNet reference CAM/eyeball เดิม แต่ยังไม่มีการเปรียบเทียบ matched samples/targets/seeds ครบทุกโมเดล ไม่ใช้ภาพสองชุดนี้อ้างว่าโมเดลหนึ่งมองดีกว่าอีกโมเดล และไม่มี ResNet eyeball comparison กับ ImageNet ที่รันเพิ่ม
5. กลุ่มduplicateที่ยืนยันไม่ข้ามsplit แต่unknownsources/undetectednear-duplicates/samesessionยังอาจมีอยู่ ภาพมีtexts/watermarksและtestsplitนี้เคยถูกดูในงานก่อนหน้า จึงยังเป็นexploratoryfixed-splitfollow-up
6. ECEใช้summary/binlogsที่มีอยู่ ไม่ได้recomputeprobabilitycalibration/testlossหรือweightedtrainlossในการตรวจซ้ำรอบรายงานนี้ ไม่ได้redecode/hashcheckpointsทุกไฟล์ใหม่
7. ตรวจกระบวนการcopiedEDA/baselineและcanonicalmanifestsแล้ว แต่ไม่ได้rerunimageauditหรือImageNetbaseline เป็นhistoricalcontextที่ใช้datasnapshot/splitเดียวกัน ไม่ใช่ผลทดลองnumericจากoldfinalruns
8. CloudW&Bvisibility/completenessยังไม่ตรวจจากdashboard;registryonlineและURLsไม่รับประกันartifactuploadครบ
9. Dropout/headstructureใช้modelsourceและinstalledtorchvisiondefaultsประกอบsavedconfigs ไม่ได้instantiateโมเดลใหม่เพื่อinspectgraph จำนวนparametersใช้savedmode_statsจริง
10. Confusion matrices หลักเป็น mean (%) ± sample SD (จุดเปอร์เซ็นต์) จาก predictions ห้า seeds ต่อคู่ ใช้สเกล 0–100 เดียวกัน; PNG seed 42 เดิมเก็บเป็นหลักฐานรอง ไม่ใช้แทนค่ารวม
11. ผู้ใช้อนุมัติและรัน augmentation/class-weight ablation ของ ResNet ครบแล้ว 10 runs; ยังไม่มี padded-input ablation, inference-latency benchmark หรือ external holdout ใหม่
12. ชื่อผู้จัดทำ/รหัสนักศึกษา/กลุ่มไม่พบในevidenceที่ใช้ จึงใช้ชื่อproject/courseแทน ไม่แต่งidentityหรือacademiccitation

## ข้อแตกต่างจากไฟล์แนบที่ตั้งใจ

- ผู้ใช้ล่าสุดขอGitHubMarkdownไม่เอาPDF: ส่งREADME+Markdownsource/inventory/validation ไม่ทำA4,PageXofYหรือPDFexport
- ไม่ยึดexampleorderingว่าDenseNetดีที่สุด: ผลจริงResNet50/FullมีmeanTestF1สูงสุด ส่วนreferenceMobileNetPartialล็อกด้วยvalidation
- ไม่ใช้Welch'stestเพราะมีmatchedseeds:คงpairedtestและplannedHolmfamilies28/30
- รูปโมเดลมาจาก latest logs/results รวม ResNet CAM ที่ผู้ใช้สร้างจาก checkpoint เดิม; ตัวอย่าง padding/augmentation สร้างจาก train images จริงและติดป้ายว่าเป็นภาพสาธิตแยกจาก measured logs ไม่มีภาพ AI หรือผลแต่ง

## วิธีทำซ้ำเฉพาะหลักฐานรายงาน

เมื่อมี raw outputs ใน workspace เดิม ใช้ PowerShell `./report/write_report.ps1` เพื่อคำนวณหลักฐานและเรียก `render_readme.js` ในขั้นสุดท้าย ข้อความของรายงานแก้ที่ `readme_template.md` ส่วน `node report/render_readme.js` อัปเดตเฉพาะ README/สำเนาและตรวจตารางกับลิงก์ ตัวสร้างหลักฐานเรียก `copy_learning_curves.ps1` เพื่อคัดลอก PNG ที่มีอยู่แล้ว พร้อมสร้าง source map และจำนวน runs ต่อ epoch; ต้องมีรูปต้นทางครบก่อนใช้คำสั่งนี้ การเผยแพร่ README ต้องรวมรูปและไฟล์ประกอบที่ลิงก์ไว้ด้วย
