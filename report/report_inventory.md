# Evidence inventory — DADS7202 Thai Shrine Classifier

ตรวจโฟลเดอร์วันที่ 19 ตุลาคม 2026 เพื่อจัดทำรายงาน Markdown ใช้ actual run artifacts เป็นหลัก ไม่ใช้ลำดับ ranking/ตัวเลขที่ยกตัวอย่างในไฟล์คำสั่งเป็นผลการทดลอง

## 1. ผลหลายรอบที่ยังคงอยู่ใน workspace

| ตำแหน่ง | สถานะ/บทบาท | ใช้ในรายงานนี้อย่างไร |
|---|---|---|
| `outputs_modes_v3_td/` | Latest protocol: 5 architectures ×3 modes ×5 tuning trials; 75 final runs, project thai-shrine-classifier-td | **แหล่งผลทดลองเพียงรอบเดียว** |
| `outputs/` | Historical Full-only screening/tuning และ25final runs | ไม่ใช้ numerical results; canonical split/EDA/baseline เดิมบางส่วนถูกคัดลอกและตรวจความเข้ากันได้ในlatest output |
| `outputs_modes_v2/` | Prepare/synthetic smoke ของprotocol12trials | ไม่รวมผลหรือcounts |
| `outputs_modes_v3/` | เตรียมภายใต้projectชื่อเดิม; registryมีincomplete attempt | ไม่รวม ไม่resume ไม่สวมเป็นผลlatest |
| `outputs_smoke/` | Smoke/debug output | ไม่รวมผลประเมินจริง |
| `wandb/` | Tracking storage ที่อยู่นอกlatest output อาจมีหลายรอบ | ไม่ใช้เป็นsourceoftruthของผลlatest |
| `dataset_cleaned/` | 560original images และmetadata | ใช้ canonical data/provenance |
| `.venv/`, `__pycache__/` | Installed runtime/bytecode cache | ไม่ใช่experiment evidence; ไม่รวมinstalled package filesในrecursive evidence CSV แต่ตรวจmodeldefaultsจากinstalledtorchvisionเพื่ออธิบายhead |

จึงตอบได้ว่า **workspace ยังมีหลายรอบอยู่จริง แต่รายงานไม่ได้เอาผลหลายรอบมาปะปนกัน** ไม่มีการลบหรือแก้raw outputsเพื่อทำรายงานนี้

## 2. การระบุ latest protocol

- Output root: `outputs_modes_v3_td/`
- Protocol fingerprint: `3a6107dd427e01f1e9d3dd3519b411450a5019505601805c1c1ebe606bf678db`
- Split CSV SHA256: `3eb9c4f3adda4a64536f809fbad5f8801a27d83265d2c753b8f98c6ae93f837f`
- Source fingerprint: `e854495869473ea313b2c455781479f1d619ac0f67c28318e8c3c01f391bd47b`
- Architectures: resnet50, efficientnet_b0, mobilenet_v3_large, densenet121, convnext_tiny
- Modes: full, frozen, partial; final seeds42/123/456/789/1024
- Reference: mobilenet_v3_large/partial; representative seed42

## 3. Recursive inventory และ run-level evidence map

[evidence_inventory.csv](evidence_inventory.csv) เก็บpath,bytes,extension,scope,architecture,seedและevidencekindของไฟล์โครงการที่ตรวจ (ไม่รวมinstalledruntime,bytecode,.gitและโฟลเดอร์reportที่สร้างเอง) snapshotประมาณ3,894files; จำนวนอาจเปลี่ยนเมื่อเพิ่มเอกสารโดยไม่เปลี่ยนผลการทดลอง

[run_evidence_map.csv](run_evidence_map.csv) ระบุrunidentity/stage/architecture/mode/seed/status/split/metricnames/raw-vs-summaryและrunpathครบ150runs เพื่อเชื่อมper-file inventoryกับความหมายทางการทดลอง Image filesที่ไม่ใช่resultsยังคงมีclass/pathในsource manifestและsplit

## 4. Map รายงานกับหลักฐาน

Pathในตารางนี้relativeต่อworkspace root; `r` หมายถึง run directory ที่ระบุในlatest `run_registry.json` ไม่ใช่folderจากhistorical outputs

| Report section / claim | Source path | Evidence type / split / identity |
|---|---|---|
| 1 datasetcounts,dimensions,formats/readerrors | outputs_modes_v3_td/dataset_audit/dataset_summary.json และ image_properties.csv | Audit summaries/individualimageproperties; copiedcompatiblecanonicalaudit |
| 1 sources/unknownURLs | outputs_modes_v3_td/source_manifest.json; dataset_cleaned/metadata_cleaned.csv | Per-image provenance; domainsนับภาพไม่ใช่independentsources |
| 1 labels/classorder | outputs_modes_v3_td/protocol.json; splits.csv; AGENTS.md mapping | Canonical4classes |
| 1 duplicatehandling | outputs_modes_v3_td/dataset_audit/duplicate_groups.csv, near_duplicate_review_decisions.csv | Reviewed integrity evidence; candidatesไม่เท่ากับconfirmedduplicate |
| 2 splitcounts/groupoverlap | outputs_modes_v3_td/splits.csv และ split_metadata.json | 560IDs/550groups; allarchitectures/modes/seeds |
| 2 preprocessing/augmentation | shrine_classifier/data.py; shrine_classifier/training.py; r/config.json | Code/config; train-onlyaugmentation |
| 3 headreplacement/weights | shrine_classifier/models.py; r/config.json; installedtorchvision0.28modelsource | Actual4-logithead; enumDEFAULT; parametercountsจากlogs |
| 3 freezingpolicies/counts | shrine_classifier_ext/finetuning_modes.py; r/config.json mode_stats | Per-architecture/mode; frozenBNpolicy |
| 3 ImageNetbaseline | outputs_modes_v3_td/baseline/baseline_metadata.json, pretrained_predictions.csv, pretrained_examples.png | QualitativeResNet50baselineบนtrain; inheritedcompatibleasset |
| 4 tuning5configs | outputs_modes_v3_td/tuning/search_definition.json | Localrandomsearch5configs reusedครบ15pairs; noSGDsample |
| 4 winnerconfigs | outputs_modes_v3_td/tuning/best_params_by_arch_mode.json | 15validation-selectedrecipes |
| 4 lock/reference/seeds | outputs_modes_v3_td/selection.json | Referenceเลือกvalidationก่อนtest |
| 4 runtime/AMP | outputs_modes_v3_td/environment.json; final r/config.json | Actualruntime; AMPfalse |
| 5/6 rawtestmetrics | final r/test_predictions.csv, run_metrics.json, test_metrics.json | Rawpersamplepredictions85/run; 75matchedruns |
| 6 learningcurves | final r/history.csv | All75runs; actualepochs; noforwardfill |
| 6 pairedstats | outputs_modes_v3_td/final/statistical_tests.csv; report/verified_statistics.csv | 28+30hypotheses; n5; joinseedID |
| 6 perclass | final r/test_predictions.csv; report/verified_per_class_summary.csv | RecomputedP/R/F1support; sampleSD |
| 6 confusionmatrices | final r/confusion_matrix_normalized.png | RepresentativeFull5modelsseed42 + validationreferenceseed42 |
| 6 calibration | outputs_modes_v3_td/confidence/run_metrics.csv, pair_summary.csv, bin_metrics.csv | Logged10binECE; calibrationไม่ได้recomputeในreportverification |
| 6 CAM/eyeball | outputs_modes_v3_td/gradcam/selected_cases.csv; eyeball/eyeball_cases.csv | Referenceonly/seed42;8selectedimages |
| 6 repeatederrors | report/error_frequency.csv | คำนวณจาก75predictionCSVs; countnotindependenttrials |
| 7 limitations/conclusions | selection.json + verifiedsummaries/stats + evidencegaps | แยกvalidationselection/testdescriptiveordering |

## 5. Figure mapping

| Report asset | Original evidence / generation |
|---|---|
| sample_grid.png, class_distribution.png, preprocessing.png | Copiedunchangedจากlatestdataset_audit/ (preprocessingต้นทางpreprocessing_examples.png) |
| image_properties.svg | Histogramคำนวณจากlatestdataset_audit/image_properties.csv;560images;overflowlastbin |
| imagenet_baseline.png | Copiedunchangedจากlatestbaseline/pretrained_examples.png;ResNet50 |
| tuning_lr.png, tuning_weight_decay.png | Copiedlatesttuning/tuning_validation_scatter.png และweight_decay_validation_scatter.png |
| all_seed_loss.svg, all_seed_acc.svg, all_seed_macro_f1.svg | 75history.csv; fivearchitecture rows × threemodecolumns;5seedcolors;solidtrain/dashedvalidation |
| pair_accuracy.png, pair_macro_f1.png | Copiedlatestreport_assets/;15pairmean/SD |
| confusion_*_full_seed42.png | finalr/confusion_matrix_normalized.png ของarchitectureนั้น/full/42 |
| confusion_reference_seed42.png | finalmobilenet_v3_large/partial/42/confusion_matrix_normalized.png |
| gradcam_reference.png | latestgradcam/reference_pair_cam_grid.png |
| eyeball_reference.png | latesteyeball/eyeball_grid.png |
| reference_reliability.png | latestconfidence/reliability_mobilenet_v3_large_partial_seed42.png |
| training_gaps.png | latestdiagnostics/training_gaps.png; supplementalasset |

`report/qa/` เป็นrasterpreviewsสำหรับตรวจSVG ไม่ใช่ผลทดลองใหม่ รูปต้นทางทั้งหมดไม่ถูกแก้ไข รูปที่สร้างเพิ่มมีเพียงEDAและall-seedcurvesจากตัวเลขที่บันทึกไว้แล้ว

## 6. Reference documents และขอบเขตคำขอ

อ่าน `CODEX_DADS7202_REPORT_INSTRUCTIONS.md` เป็นแนวทางโครงสร้าง/technicalexpectations **ไม่ใช้ตัวเลขหรือrankingที่ยกตัวอย่างเป็นevidence** ไม่พบ `HOMEWORK_GUIDELINE (1).pdf` และ `HOMEWORK_Example.pdf` จากการค้นโครงการ (ยกเว้นruntime) จึงไม่ยืนยันvisualstyleหรือข้อกำหนดต้นฉบับPDF คำขอผู้ใช้ล่าสุดให้ทำMarkdown/GitHubREADMEจึงแทนที่PDF/A4/pagenumberingในไฟล์แนบ

Canonicalreportคือ [README.md](../README.md); [report_source.md](report_source.md) เป็นสำเนาที่ปรับrelativepathsสำหรับอ่านจากโฟลเดอร์report ไม่ใช่protocolหรือผลทดลองอีกversionหนึ่ง
