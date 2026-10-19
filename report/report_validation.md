# Report validation — 19 October 2026

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
- Confusion matrices ของทุก 15 คู่เป็นเปอร์เซ็นต์ต่อ true-class row แล้วคำนวณ mean/sample SD ข้าม 5 seeds มีสเกลสี 0–100 เดียวกัน สีแยกตาม mode และ [ค่ารายช่อง](confusion_mean_sd.csv)
- สี/สัญลักษณ์ในกราฟ mode คงที่: Frozen น้ำเงิน/วงกลม, Partial แดง/สี่เหลี่ยม, Full เขียว/ข้าวหลามตัด ส่วนสีของคลาสใช้เฉพาะ EDA
- ข้อมูล [brightness](derived_brightness.csv) คำนวณจากภาพต้นฉบับ 560 ภาพด้วย Sharp, EXIF-aware decode และ Rec.709 luminance; ไม่เอาตัวเลขจากภาพตัวอย่างของผู้ใช้
- เพิ่มขั้นตอนเลือก reference ก่อน Grad-CAM, confidence/calibration และระบุสถานะ ablation ที่ยังไม่รันอย่างชัดเจน
- เพิ่มภาพ padding ขนาดใหญ่จากภาพ train จริงหนึ่งภาพ และกริด augmentation 4 คลาส × 5 ภาพจาก canonical train split; เก็บ IDs, SHA256, seeds และพารามิเตอร์ใน [preprocessing_preview_sources.json](preprocessing_preview_sources.json) โดยระบุใน README ว่าเป็นภาพสาธิตที่สร้างใหม่ ไม่ใช่ minibatch ใน log
- แก้ตารางสัดส่วนข้อมูลที่มี `%` แยกเป็นอีกคอลัมน์ ตารางสถิติที่ comparison แตกเป็นหลาย cells และตารางรายคลาสที่ค่ารวมอยู่ cell เดียว แก้การจัดกลุ่ม expression ในตัวสร้างตารางและเพิ่มการตรวจจำนวน cells ก่อนเขียนไฟล์
- ย่อตารางสูตรชนะให้อ้าง trial ในตาราง search configurations และปรับภาษาเนื้อหารายงาน โดยคงค่าผลทดลองและข้อสรุปทางสถิติเดิม
- การปรับรอบนี้ใช้ PowerShell อ่านผลเดิมและ Node.js สร้างกราฟ/ตรวจลิงก์ ไม่ได้รัน Python, training หรือชุดทดสอบโค้ด

## ข้อจำกัด/สิ่งที่ยังไม่ยืนยัน

1. **[TODO — source evidence not found]** PDF guideline/exampleต้นฉบับ ไม่มีในโครงการที่ค้น จึงใช้7sectionorderจากinstructionMDและคำขอล่าสุดที่ให้MarkdownแทนPDF ไม่อ้างว่าตรวจPDFแล้ว
2. **[TODO — source evidence not found]** Collection/scraperprotocollogที่บอกquery/date/manual-vs-scrapedแต่ละภาพและlicenseครบทุกภาพ ไม่มีหลักฐานพอ รายงานใช้URLmetadataที่มีจริงและระบุunknown112ภาพ
3. Brightness values ไม่อยู่ใน audit เดิม จึงคำนวณใหม่จากภาพต้นฉบับด้วยวิธีที่บันทึกใน `generate_latest_figures.js` และเก็บค่ารายภาพไว้ใน `derived_brightness.csv`; ไม่ตีความเป็นผลของการฝึก
4. Cross-modelGrad-CAMบนmatchedsamplesทั้ง5architecturesไม่มี มีreferenceCAM/eyeballจริงเท่านั้น ไม่สร้างattentioncomparisonขึ้นเองและไม่รันcheckpointinferenceเพิ่ม
5. กลุ่มduplicateที่ยืนยันไม่ข้ามsplit แต่unknownsources/undetectednear-duplicates/samesessionยังอาจมีอยู่ ภาพมีtexts/watermarksและtestsplitนี้เคยถูกดูในงานก่อนหน้า จึงยังเป็นexploratoryfixed-splitfollow-up
6. ECEใช้summary/binlogsที่มีอยู่ ไม่ได้recomputeprobabilitycalibration/testlossหรือweightedtrainlossในการตรวจซ้ำรอบรายงานนี้ ไม่ได้redecode/hashcheckpointsทุกไฟล์ใหม่
7. ตรวจกระบวนการcopiedEDA/baselineและcanonicalmanifestsแล้ว แต่ไม่ได้rerunimageauditหรือImageNetbaseline เป็นhistoricalcontextที่ใช้datasnapshot/splitเดียวกัน ไม่ใช่ผลทดลองnumericจากoldfinalruns
8. CloudW&Bvisibility/completenessยังไม่ตรวจจากdashboard;registryonlineและURLsไม่รับประกันartifactuploadครบ
9. Dropout/headstructureใช้modelsourceและinstalledtorchvisiondefaultsประกอบsavedconfigs ไม่ได้instantiateโมเดลใหม่เพื่อinspectgraph จำนวนparametersใช้savedmode_statsจริง
10. ConfusionPNGเดิมrow-normalizedเป็น0–1แต่colorbarอาจautoscaleต่างขอบกัน จึงแจ้งให้เปรียบเทียบcellnumbersไม่ใช้colorintensityข้ามรูปเพียงอย่างเดียว ไม่ได้สร้างaverageconfusionใหม่
11. ไม่มีoptionalaugmentation/classweight/paddedablation,ไม่มีinference-latencybenchmarkและไม่มีexternalholdoutใหม่ ไม่เพิ่มการทดลองนอกงบเพื่อเติมรายงาน
12. ชื่อผู้จัดทำ/รหัสนักศึกษา/กลุ่มไม่พบในevidenceที่ใช้ จึงใช้ชื่อproject/courseแทน ไม่แต่งidentityหรือacademiccitation

## ข้อแตกต่างจากไฟล์แนบที่ตั้งใจ

- ผู้ใช้ล่าสุดขอGitHubMarkdownไม่เอาPDF: ส่งREADME+Markdownsource/inventory/validation ไม่ทำA4,PageXofYหรือPDFexport
- ไม่ยึดexampleorderingว่าDenseNetดีที่สุด: ผลจริงResNet50/FullมีmeanTestF1สูงสุด ส่วนreferenceMobileNetPartialล็อกด้วยvalidation
- ไม่ใช้Welch'stestเพราะมีmatchedseeds:คงpairedtestและplannedHolmfamilies28/30
- รูปทั้งหมดมาจากlatestlogsหรือcompatiblecanonicalEDA/baseline;รูปใหม่เป็นการplotrecordednumbersเท่านั้น ไม่มีภาพAIหรือผลแต่ง

## วิธีทำซ้ำเฉพาะหลักฐานรายงาน

เมื่อมี raw outputs ใน workspace เดิม ใช้ PowerShell `./report/write_report.ps1` เพื่อคำนวณหลักฐานและอัปเดต generated table blocks โดยเก็บข้อความใน README ไว้ ตัวสร้างรายงานเรียก `copy_learning_curves.ps1` เพื่อคัดลอก PNG ที่มีอยู่แล้ว พร้อมสร้าง source map และจำนวน runs ต่อ epoch; ต้องมีรูปต้นทางครบก่อนใช้คำสั่งนี้ การเผยแพร่ README ต้องรวมรูปและไฟล์ประกอบที่ลิงก์ไว้ด้วย
