# Evidence inventory — DADS7202 Thai Shrine Classifier

ตรวจโฟลเดอร์วันที่ 19 ตุลาคม 2026 เพื่อจัดทำรายงาน Markdown ใช้ actual run artifacts เป็นหลัก ไม่ใช้ลำดับ ranking/ตัวเลขที่ยกตัวอย่างในไฟล์คำสั่งเป็นผลการทดลอง

## รูปแบบรายงานที่ปรับเมื่อ 20 ตุลาคม 2026

การแก้ครั้งนี้เปลี่ยนการนำเสนอ ไม่เพิ่มผลการทดลอง ผลเชิงตัวเลขและต้นฉบับภาพที่บันทึกไว้ยังคงเดิม ตารางแสดงเปอร์เซ็นต์/pp ผ่าน `render_readme.js` และใช้ `readme_template.md` เป็นแม่แบบของ README ส่วน `report_source.md` เป็นสำเนารายงานเดียวกัน

| ภาพที่ใช้ใน README | ต้นทางและการปรับ |
| --- | --- |
| `latest/imagenet_baseline_percent.png` | ImageNet predictions เดิมของ train และภาพเดิม; แสดงสามตัวอย่างแรกที่บันทึกไว้ต่อคลาส พร้อม confidence (%) |
| `latest/gradcam_resnet50_full_seed42_readable.png` | แปด cases เดิมของ ResNet; จัดวางใหม่จาก original/heatmap/overlay โดยไม่ inference หรือสร้าง heatmap ใหม่ |
| `latest/resnet_ablation_summary.svg` | Baseline final ห้า seeds และ ablation สิบ runs เดิม; ตรวจ mean/SD ตรงกับ saved summary |
| `latest/gradcam_reference_percent.png`, `latest/eyeball_reference_percent.png` | MobileNet reference แปด cases เดิม; จัดรูปและป้าย confidence (%) ใหม่ เก็บเป็นภาคผนวก |
| กราฟ EDA, tuning, checkpoint, mode, confusion และ calibration ใน `latest/` | เปลี่ยน theme จาก saved audit/metrics/predictions/bin logs; คง class order และใช้ confusion heat scale เดียวกันทุกคู่ |

ที่มาและ SHA256 ของภาพที่จัดใหม่อยู่ใน [followup_figure_sources.json](followup_figure_sources.json), [context_figure_sources.json](context_figure_sources.json) และ [latest_figure_sources.json](latest_figure_sources.json) การตรวจรูปแบบอยู่ใน [qa/presentation_validation.json](qa/presentation_validation.json) ส่วนการเทียบหลักฐานเดิม 86 ไฟล์อยู่ใน [qa/evidence_preservation.json](qa/evidence_preservation.json) ภาพต้นฉบับของรอบก่อนปรับรูปแบบยังเก็บไว้และไม่ใช้แทนผลใหม่

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

ไฟล์ `report/evidence_inventory.csv` ใน workspace เก็บ path, bytes, extension, scope, architecture, seed และชนิดหลักฐานของไฟล์ที่ตรวจ ไม่รวม installed runtime, bytecode, `.git` และโฟลเดอร์รายงาน Snapshot เดิมมีประมาณ 3,894 ไฟล์ จำนวนอาจเปลี่ยนเมื่อเพิ่มเอกสารหรือกราฟ

ไฟล์ `report/run_evidence_map.csv` ใน workspace ระบุ identity, stage, architecture, mode, seed, status และ run path ของงานหลัก 150 runs ส่วน ablation ใหม่ 10 runs มี registry แยกใน `outputs_modes_v3_td/ablation/resnet50_full_v1/run_registry.json` จึงมีการฝึกทั้งหมด 160 runs โดยไม่ได้นับ baseline ที่ reuse ซ้ำ [learning_curve_sources.json](learning_curve_sources.json) ระบุ histories และ SHA256 ของทั้ง 75 final runs รวมถึงกราฟที่นำมาใช้

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
| 6 learning curves | final r/history.csv; outputs_modes_v3_td/final/curves/; report/learning_curve_sources.json | 75 runs; mean และ sample SD ต่อ epoch เฉพาะ runs ที่ยังมีข้อมูล ไม่มี forward fill; จำนวน runs ต่อ epoch อยู่ใน learning_curve_epoch_counts.csv |
| 6 pairedstats | outputs_modes_v3_td/final/statistical_tests.csv; report/verified_statistics.csv | 28+30hypotheses; n5; joinseedID |
| 6 perclass | final r/test_predictions.csv; report/verified_per_class_summary.csv | RecomputedP/R/F1support; sampleSD |
| 6 confusionmatrices | final r/test_predictions.csv; report/confusion_mean_sd.csv | ทุก 15 คู่: per-seed row percentages → mean/sample SD ข้าม 5 seeds; fixed 0–100 colour scale |
| 6 calibration | outputs_modes_v3_td/confidence/run_metrics.csv, pair_summary.csv, bin_metrics.csv | Logged10binECE; calibrationไม่ได้recomputeในreportverification |
| 6 CAM/eyeball | outputs_modes_v3_td/gradcam/selected_cases.csv; eyeball/eyeball_cases.csv | Referenceonly/seed42;8selectedimages |
| 6 repeatederrors | report/error_frequency.csv | คำนวณจาก75predictionCSVs; countnotindependenttrials |
| 7 limitations/conclusions | selection.json + verifiedsummaries/stats + evidencegaps | แยกvalidationselection/testdescriptiveordering |

## 5. Figure mapping

| Report asset | Original evidence / generation |
|---|---|
| latest/*.svg | README ฉบับปัจจุบัน: สร้างจาก saved logs/predictions/audit ด้วย `generate_latest_figures.js` และระบุ SHA256 ใน `latest_figure_sources.json` |
| latest/selected_resnet50_full_calibration.svg | สร้างจาก saved `confidence/bin_metrics.csv`, ResNet50/Full seed 42 ด้วย `generate_selected_model_calibration.js`; source hash และการเลือกโมเดลในรายงานอยู่ใน `selected_model_report_sources.json` |
| latest/padding_large.png, latest/augmentation_train_5_per_class.png | ตัวอย่างประกอบจาก original images ใน canonical train split สร้างด้วย `generate_preprocessing_examples.js`; แหล่งภาพและค่าการแปลงอยู่ใน `preprocessing_preview_sources.json` ไม่ใช่ภาพแนบหรือ logged training minibatch |
| sample_grid.png, class_distribution.png, preprocessing.png | Copiedunchangedจากlatestdataset_audit/ (preprocessingต้นทางpreprocessing_examples.png) |
| image_properties.svg | Histogramคำนวณจากlatestdataset_audit/image_properties.csv;560images;overflowlastbin |
| imagenet_baseline.png | Copiedunchangedจากlatestbaseline/pretrained_examples.png;ResNet50 |
| tuning_lr.png, tuning_weight_decay.png | Copiedlatesttuning/tuning_validation_scatter.png และweight_decay_validation_scatter.png |
| learning_curves/<architecture>/mean_sd_grid.png | คัดลอกโดยไม่แก้ไขจาก outputs_modes_v3_td/final/curves/<architecture>/mean_sd_grid.png ครบ 5 สถาปัตยกรรม; loss/accuracy ของ Full, Frozen, Partial |
| learning_curves/mobilenet_v3_large/partial/mean_sd_curves.png | คัดลอกจากคู่โมเดลอ้างอิง; loss, accuracy และ Macro F1 |
| learning_curves/all_pairs_grid.png | คัดลอก overview ทั้ง 15 คู่; ลิงก์ภาพขนาดเต็มใน README |
| pair_accuracy.png, pair_macro_f1.png | Copiedlatestreport_assets/;15pairmean/SD |
| confusion_*_full_seed42.png | finalr/confusion_matrix_normalized.png ของarchitectureนั้น/full/42 |
| confusion_reference_seed42.png | finalmobilenet_v3_large/partial/42/confusion_matrix_normalized.png |
| gradcam_reference.png | latestgradcam/reference_pair_cam_grid.png |
| eyeball_reference.png | latesteyeball/eyeball_grid.png |
| reference_reliability.png | latestconfidence/reliability_mobilenet_v3_large_partial_seed42.png |
| training_gaps.png | latestdiagnostics/training_gaps.png; supplementalasset |

กราฟ learning curves เดิมที่ใช้เป็นหลักฐานประกอบมี 7 ไฟล์ คัดลอกแบบ byte-identical และตรวจ SHA256 เทียบต้นทางแล้ว แต่ละคู่เริ่มจาก 5 seeds; หลัง early stopping จำนวนที่ใช้เฉลี่ยลดลงตามข้อมูลที่เหลือ Script ต้นทางใช้ `nanmean` และ `nanstd(ddof=1)` และแทน SD ด้วย 0 เมื่อเหลือ run เดียว ซึ่งเป็นวิธีแสดงผล ไม่ใช่หลักฐานว่าความแปรปรวนเป็นศูนย์ ดูจำนวน runs จริงใน [learning_curve_epoch_counts.csv](learning_curve_epoch_counts.csv)

SVG `all_seed_loss.svg`, `all_seed_acc.svg`, `all_seed_macro_f1.svg` และ previews ใน `report/qa/` เป็นหลักฐานรองจากรายงานฉบับก่อน README ปัจจุบันใช้ box plots ของ checkpoint และกราฟเปรียบเทียบ modes ใน `report/report_assets/latest/`; mean ± SD temporal curves เดิมยังเข้าถึงได้ทางลิงก์ รูปต้นทางใน output ไม่ถูกแก้ไข และไม่ได้รัน Python เพื่อสร้างกราฟใหม่

กราฟใน `report/report_assets/latest/` สร้างด้วย [generate_latest_figures.js](generate_latest_figures.js): class/dimension/colour จาก `outputs_modes_v3_td/dataset_audit/image_properties.csv`; brightness จากภาพต้นฉบับทั้ง 560 ภาพและ [derived_brightness.csv](derived_brightness.csv); sweep จาก `tuning/all_trials.csv`; mode comparison/box plots จาก `final/run_metrics.csv`; confusion 15 คู่จาก `final/**/test_predictions.csv`; calibration จาก `confidence/bin_metrics.csv` SHA256 ของ CSV ต้นทางอยู่ใน [latest_figure_sources.json](latest_figure_sources.json) ไม่มีภาพแนบตัวอย่างของผู้ใช้เป็น source data

ภาพตัวอย่าง preprocessing สองภาพสร้างจาก original train files แยกจากผล measured runs โดยใช้ neutral letterbox ที่ขนาด 224 ตามโค้ด และแสดง augmentation ตามช่วงค่าที่กำหนดใน `shrine_classifier/training.py` การเรนเดอร์ตัวอย่างใช้ JavaScript จึงไม่อ้างว่าเป็น output แบบ bit-identical จาก torchvision หรือเป็นภาพที่ log ระหว่างฝึก ดู source IDs และ parameters ใน [preprocessing_preview_sources.json](preprocessing_preview_sources.json)

## 6. Reference documents และขอบเขตคำขอ

อ่าน `CODEX_DADS7202_REPORT_INSTRUCTIONS.md` เป็นแนวทางโครงสร้าง/technicalexpectations **ไม่ใช้ตัวเลขหรือrankingที่ยกตัวอย่างเป็นevidence** ไม่พบ `HOMEWORK_GUIDELINE (1).pdf` และ `HOMEWORK_Example.pdf` จากการค้นโครงการ (ยกเว้นruntime) จึงไม่ยืนยันvisualstyleหรือข้อกำหนดต้นฉบับPDF คำขอผู้ใช้ล่าสุดให้ทำMarkdown/GitHubREADMEจึงแทนที่PDF/A4/pagenumberingในไฟล์แนบ

Canonicalreportคือ [README.md](../README.md); [report_source.md](report_source.md) เป็นสำเนาที่ปรับrelativepathsสำหรับอ่านจากโฟลเดอร์report ไม่ใช่protocolหรือผลทดลองอีกversionหนึ่ง

## 7. การเลือกโมเดลในรายงาน — 20 October 2026

คำขอล่าสุดให้เลือก ResNet50/Full จากผลประเมินหลายด้านหลังดู test เป็นข้อสรุปเดียวของรายงาน พร้อมเปิดเผย post-hoc selection และความจำเป็นของ external holdout ใหม่ การตัดสินใจนี้ไม่ได้แก้ `outputs_modes_v3_td/selection.json` หรือ reference/families ของ paired tests เดิม Confusion matrix, Grad-CAM และ calibration หลักใช้ ResNet50/Full; CAM/eyeball ของ MobileNet อยู่ในส่วนหลักฐานประกอบที่ติดป้ายชัดเจน

ผู้ใช้รัน Python follow-up เสร็จแล้วในวันที่ 20 October: Grad-CAM ของ ResNet50/Full seed 42 มี 8 cases (ถูก 6, ผิด 2) และ ablation ครบ 2 variants × 5 seeds ใช้ 145 epochs โดย reuse baseline เดิม 5 seeds รวมการฝึกทั้งงาน 2,906 epochs Agent ตรวจไฟล์/คำนวณซ้ำด้วย Node โดยไม่ได้รัน Python/inference/training เพิ่ม ผล ablation ไม่มี comparisons ที่ significant หลัง Holm ทั้ง 4 แถว

| Follow-up report artifact | Source |
| --- | --- |
| `report_assets/latest/gradcam_resnet50_full_seed42.png`, `resnet_gradcam_cases.csv` | `outputs_modes_v3_td/gradcam/resnet50_full/seed42/20261020T133536Z_f92b166e/gradcam_grid.png`, `selected_cases.csv` |
| `report_assets/latest/resnet_ablation_summary.png` | `outputs_modes_v3_td/ablation/resnet50_full_v1/ablation_summary.png` |
| `resnet_ablation_*.csv` | Same study directory: condition summary, matched deltas, per-class summary, statistical tests |
| `resnet_followup_sources.json` | Exported manifest: source directories, registry hash, checkpoint and output hashes |
| `resnet_followup_verification.json` | Independent Node recomputation from 15 × 85 predictions, summary rows and four statistical comparisons |

รูป/ตารางสำเนาตรงกับ SHA256 ของต้นทาง Grad-CAM/ablation ที่สร้างโดยผู้ใช้ ปัจจุบันส่วนหลักของ README ใช้ ResNet สำหรับ confusion matrix, Grad-CAM และ calibration ส่วน MobileNet CAM/eyeball เป็นหลักฐานประกอบของ reference เดิม การตรวจนี้ไม่ยืนยันความครบถ้วนของ W&B cloud artifacts
