# Report validation — 19 October 2026

## สิ่งที่ตรวจแล้ว

- ใช้resultsเฉพาะ `outputs_modes_v3_td` และตรวจว่ามีhistoricaloutputอื่นในworkspaceจริง โดยคัดออกอย่างชัดเจน ไม่มีการลบ/เขียนทับrawexperimentoutputs
- ครบ75uniquefinalrunidentitiesและ15pairsคู่ละ5matchedseeds; canonicaltestorderตรงทุกrun (85ภาพ/run)
- คำนวณAccuracy,macroP/R/F1และper-classP/R/F1/supportใหม่จากpredictionCSVsรวม6,375rows; zero_division=0
- คำนวณmean/sampleSD(ddof1)ใหม่ทั้ง15pairsและ60classrows ไม่ใช้screenshotหรือmaximumseedเป็นหลักฐานแทน
- ตรวจgroupsplitoverlapจากsplits.csv=0; ไม่ถือว่าครอบคลุมsource/sessionleakageที่ไม่ทราบ
- Pairedseedjoinsตรงกันและไม่พบduplicatehypotheses; independentlyrecomputedt/rawpและHolm28+30comparisons
- เพิ่มCohen'sd_zและ95%CIของpairedmeanโดยไม่เลือกcomparisonfamiliesใหม่จากผลtest
- ตรวจรูปSVGใหม่ว่าXMLparseได้,renderด้วยNode/sharpและดูภาพจริงสำหรับloss/F1/EDA; Accuracyใช้renderer/axesโครงสร้างเดียวกันและXMLผ่าน กราฟชุดเดิมคัดลอกโดยไม่ดัดแปลง
- READMEมีหัวข้อหลัก7ส่วนตามลำดับ,unitsของAccuracyเป็น%และF1/P/Rเป็น0–1,ระบุmode/seed/aggregationไว้ใต้รูป และไม่สรุปว่าnon-significant=equal
- MainREADMEและreport_sourceใช้relativeassetpaths;ไม่มีPDFและไม่มีการpublishGitHubแทนผู้ใช้

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

## ข้อจำกัด/สิ่งที่ยังไม่ยืนยัน

1. **[TODO — source evidence not found]** PDF guideline/exampleต้นฉบับ ไม่มีในโครงการที่ค้น จึงใช้7sectionorderจากinstructionMDและคำขอล่าสุดที่ให้MarkdownแทนPDF ไม่อ้างว่าตรวจPDFแล้ว
2. **[TODO — source evidence not found]** Collection/scraperprotocollogที่บอกquery/date/manual-vs-scrapedแต่ละภาพและlicenseครบทุกภาพ ไม่มีหลักฐานพอ รายงานใช้URLmetadataที่มีจริงและระบุunknown112ภาพ
3. **[TODO — source evidence not found]** Brightnessdistribution ไม่มีbrightnessvaluesในauditที่ใช้ ไม่สร้างค่าที่ไม่มีหลักฐาน;แทนด้วยwidth/height/aspect/filesizeplots
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

เมื่อมีrawoutputsอยู่ในworkspaceเดิม สามารถใช้PowerShell `./report/write_report.ps1` เพื่อคำนวณverification/assets/tablesใหม่ โดยไม่ใช้Python/training MainREADMEข้อความอธิบายถูกเก็บไว้และอัปเดตเฉพาะgeneratedtableblocks ก่อนpublishต้องcommitREADMEและโฟลเดอร์reportรวมassetsด้วย ไม่จำเป็นต้องcommitcheckpointsหรือW&Bbinaryfiles
