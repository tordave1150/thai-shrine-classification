# สรุปการทดลอง Thai Shrine Classifier ตั้งแต่ข้อมูลจนถึงผลลัพธ์

จัดทำ 14 ตุลาคม 2026 จากผลที่บันทึกใน `outputs_modes_v3_td` โดยอ่าน CSV/JSON และใช้ภาพกราฟเดิม ไม่มีการรัน Python หรือฝึกเพิ่มเพื่อทำเอกสารนี้ ตัวเลขผลใหม่ไม่ได้นำมาปะปนกับผล Full-only ใน `outputs/` หรือ output ที่เตรียมไว้ก่อนเปลี่ยน W&B project

## 1. งานนี้ต้องการตอบอะไร

สร้างโมเดลจำแนกภาพศาล 4 ประเภท และเปรียบเทียบ CNN ที่เรียนจาก ImageNet มาแล้ว 5 แบบ ได้แก่ ResNet50, EfficientNet-B0, MobileNetV3-Large, DenseNet121 และ ConvNeXt-Tiny ในการฝึก 3 mode: Full, Frozen และ Partial รวม 15 คู่

คำถามหลักคือ เมื่อ **แต่ละ architecture × mode มี tuning ของตัวเองในงบเท่ากัน** คู่ใดให้ผลดีและมีความสม่ำเสมอ ไม่เลือกสูตรจาก Full แล้วเอาไปใช้กับ Frozen/Partial โดยไม่ได้ค้นหาสูตรของ mode นั้น

ผลหลักที่ควรรู้:

- จบ 75 tuning trials และ 75 final runs รวม 150 runs; final ทุกคู่มี 5 seeds
- คู่ที่เลือกจาก validation ก่อนประเมิน test คือ **MobileNetV3-Large / Partial**: tuning validation Macro F1 = 97.34%, Accuracy = 97.65%
- คู่ที่มีค่าเฉลี่ย test Macro F1 สูงสุดจากผลที่สังเกตคือ **ResNet50 / Full**: 92.56 ± 1.81%; test Accuracy = 93.65 ± 1.34%
- Reference MobileNetV3-Large / Partial มี test Macro F1 = 91.68 ± 3.05% และ Accuracy = 93.18 ± 2.41% ไม่เปลี่ยน reference ย้อนหลังเพราะเห็น test ranking
- Frozen มีค่าเฉลี่ย test F1 ต่ำกว่า Full/Partial ของ architecture เดียวกันทั้งห้าแบบ แต่ไม่ใช่ทุกผลต่างจะมีนัยสำคัญหลังปรับ Holm

เครื่องหมาย ± ในผล test หมายถึง sample SD ของ 5 training seeds ไม่ใช่ confidence interval หรือความไม่แน่นอนบน dataset ใหม่

## 2. แผนตั้งแต่เริ่มจนจบ

```text
ข้อมูลเดิมและ audit → grouped split เดิม → preprocessing/augmentation
→ ImageNet baseline เชิงคุณภาพ → ตรวจ implementation/smoke
→ tuning แยก 15 คู่ คู่ละ 5 trials (train + validation เท่านั้น)
→ lock สูตรชนะ 15 คู่ + reference + seeds
→ final ฝึกใหม่จาก ImageNet 15 คู่ × 5 seeds
→ เลือก checkpoint ด้วย validation แล้ว test ครั้งเดียวต่อ run
→ สรุป mean/SD + paired tests + calibration + training dynamics
→ Grad-CAM/eyeball ที่ reference seed 42 → HTML และ Markdown นี้
```

รอบล่าสุดใช้ขั้น `prepare, smoke, tune, lock, final, analyze, gradcam, report` ของ `pipeline_modes.py` โดย `gradcam` สร้าง eyeball ด้วย การ audit/split และ baseline บางส่วนใช้ artifacts เดิมที่ตรวจและคัดลอกมายัง output ใหม่ ไม่ได้แอบอ้างว่าเพิ่งสร้างใหม่ทุกไฟล์ รอบล่าสุดไม่ได้ทำ common-recipe screening เพิ่ม และไม่ได้รัน optional no-augmentation/no-class-weight/padded-input ablation

## 3. ข้อมูลและการป้องกัน leakage

ใช้ `dataset_cleaned` ภาพต้นฉบับก่อนขั้น padding ของโครงการ ไม่เอาภาพ cleaned และ padded มารวมเป็นตัวอย่างคนละภาพ

| Class | ความหมาย | ทั้งหมด | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| san_phraphrom | ศาลพระพรหม | 103 | 70 | 17 | 16 |
| san_phraphum | ศาลพระภูมิ | 124 | 89 | 17 | 18 |
| san_tayay | ศาลตายาย | 133 | 91 | 21 | 21 |
| tee_ju_eia | ศาลตี่จู้เอี๊ยะ | 200 | 140 | 30 | 30 |
| รวม | | 560 | 390 | 85 | 85 |

มี 550 independent groups ภาพซ้ำ/แหล่งภาพเดียวกันอยู่ group เดียวก่อนแบ่งชุด จึงไม่ให้ group ข้าม train/validation/test ใช้ canonical split seed 42 เดิม ทุก architecture/mode/seed ใช้ภาพในแต่ละชุดตรงกัน Training seed เปลี่ยนการฝึก แต่ไม่เปลี่ยน split

ตรวจความสมบูรณ์ของไฟล์และ metadata รวมถึง actual format, RGB/alpha/CMYK, dimensions, byte hash, decoded-pixel hash และ provenance ภาพมี URL ทราบแหล่ง 448 ภาพ และไม่ทราบ 112 ภาพ; ไม่ทราบ URL ไม่ได้แปลว่าภาพเป็นอิสระจากกันแน่นอน กลุ่ม duplicate และการ review เดิมถูกเก็บไว้ ไม่ลบภาพเดิมเพื่อให้ตัวเลขดูดี

หลักฐาน: [manifest](C:/Users/User/Desktop/deep/outputs_modes_v3_td/source_manifest.json), [split](C:/Users/User/Desktop/deep/outputs_modes_v3_td/splits.csv), [split metadata](C:/Users/User/Desktop/deep/outputs_modes_v3_td/split_metadata.json)

![จำนวนภาพแยก class](C:/Users/User/Desktop/deep/outputs_modes_v3_td/dataset_audit/class_distribution.png)

กราฟ distribution ใช้อ่านความไม่สมดุลของข้อมูล: ศาลตี่จู้มีภาพมากกว่าศาลพระพรหม ไม่ควรดู accuracy อย่างเดียวจนละเลย class ที่มีภาพน้อย

![ตัวอย่างข้อมูล](C:/Users/User/Desktop/deep/outputs_modes_v3_td/dataset_audit/sample_grid.png)

![Duplicate review เดิม](C:/Users/User/Desktop/deep/outputs_modes_v3_td/dataset_audit/near_duplicate_review.png)

ภาพ review ใช้ตรวจความใกล้เคียงและประกอบการจัดกลุ่ม ไม่ใช่คะแนนโมเดล และ perceptual similarity candidate ไม่ได้หมายความว่าเป็น duplicate ที่ยืนยันแล้วทุกคู่

## 4. Preprocessing, augmentation และ class weights

Decode ตาม format จริง → EXIF transpose → แปลง RGB อย่างสม่ำเสมอและ composite transparency → resize รักษาสัดส่วน → วางกลาง canvas 224×224 สี neutral `(124,116,104)` → tensor → ImageNet normalization mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` ไม่ยืดภาพเป็นสี่เหลี่ยมและไม่ตัด dark content อัตโนมัติ

Train เท่านั้นใช้ rotation ±12°, translation ±5%, scale .92–1.08, perspective .10 ด้วยโอกาส .20, horizontal flip .5, color jitter .10/.10/.10 และ hue .02, Random Erasing .05 ด้วยพื้นที่ .02–.08 และ neutral fill ส่วน validation/test deterministic ไม่มี random augmentation

![ก่อนและหลัง preprocessing](C:/Users/User/Desktop/deep/outputs_modes_v3_td/dataset_audit/preprocessing_examples.png)

ภาพนี้ช่วยดูว่าศาลยังคงสัดส่วนและเนื้อหาไว้ แต่ neutral padding ไม่ได้พิสูจน์ว่าโมเดลจะไม่ใช้ background/aspect ratio เป็น shortcut

ใช้ weighted cross entropy จาก train counts เท่านั้น: `weight[c] = 390 / (4 × train_count[c])` ได้ weights ประมาณ `[1.392857, 1.095506, 1.071429, 0.696429]` ทำให้ class ที่ train น้อยมีน้ำหนักมากขึ้น ไม่ใช้ test มาคำนวณ weight และไม่เพิ่ม oversampling ซ้อนกัน

## 5. ImageNet baseline และการเปลี่ยน classifier

ใช้ pretrained torchvision ImageNet weights แล้วเปลี่ยน final output layer เป็น 4 logits ส่วน baseline ใช้โมเดล ImageNet ที่ยังไม่เปลี่ยน label space เพื่อแสดงว่าป้าย ImageNet 1,000 classes ไม่ตรงกับชื่อศาลไทย ไม่คำนวณ four-class accuracy ให้ baseline นี้

![ImageNet baseline ตัวอย่างเดิม](C:/Users/User/Desktop/deep/outputs_modes_v3_td/baseline/pretrained_examples.png)

ผล ImageNet เช่น stupa, altar หรือ china cabinet สามารถสัมพันธ์กับองค์ประกอบภาพได้ แต่ยังไม่ใช่คำตอบของโจทย์ 4 classes จึงต้อง fine-tune

## 6. Full/Frozen/Partial ทำต่างกันอย่างไร

| Mode | สิ่งที่ฝึก | สิ่งที่คงไว้ |
|---|---|---|
| Full | Backbone และ classifier ทั้งหมด | เริ่มจาก pretrained weights แล้วปรับทั้งหมด |
| Frozen | Classifier head | Backbone frozen และ eval รวมถึง BatchNorm buffers |
| Partial | Final semantic backbone stage และ classifier | Backbone ส่วนต้น frozen/eval |

Partial ของ ResNet ใช้ layer4; DenseNet ใช้ denseblock4/norm5; ConvNeXt ใช้ final feature stage; EfficientNet ใช้ final MBConv stage/projection; MobileNet ใช้ final semantic group/projection ใน logs reference คือ `features.13–16` และ `classifier` MobileNet Partial reference มี trainable 3,414,668 และ frozen 792,488 parameters

การเลือก mode เกิด **ก่อนสร้าง optimizer ในทุก trial และ final run** ไม่ใช่เอา freeze/unfreeze มาทดลองทีหลังบนสูตร Full เดียว การเปรียบเทียบจึงให้ทุกคู่มีโอกาสค้นสูตรเท่ากัน อย่างไรก็ดี 5 trials ยังเป็นการค้นขนาดเล็ก ไม่รับประกัน optimum

## 7. Tuning: sweep/search ทำอะไรและได้อะไร

แต่ละคู่ค้น learning rate, weight decay, optimizer และ batch size แล้วฝึกด้วย seed 42 ใช้ train เพื่อปรับ weights และ validation เพื่อเลือก checkpoint/สูตร ไม่ใช้ test

Search space: LR log-uniform `[1e-5,1e-3]`, weight decay `[1e-6,1e-2]`, optimizer `{adamw,adam,sgd}`, batch `{16,32}`

**วิธีที่รันจริงเป็น local seeded random search ส่ง trial แต่ละตัวไป W&B online ไม่ใช่ W&B cloud Sweep agent** ใช้ 5 sampled configurations ชุดเดียวกันทดสอบครบทุกคู่ เพื่อให้ search candidates เท่ากัน แต่แต่ละคู่เลือกผู้ชนะของตัวเอง ไม่ inherit Full winner

| Trial | LR | Weight decay | Optimizer | Batch |
| --- | --- | --- | --- | --- |
| 1 | 0.0003531117 | 5.695263E-05 | adam | 32 |
| 2 | 0.0002481624 | 2.380726E-06 | adam | 32 |
| 3 | 0.0003328736 | 0.001393982 | adam | 16 |
| 4 | 7.957413E-05 | 3.04223E-05 | adamw | 32 |
| 5 | 0.0001939681 | 0.001954549 | adam | 16 |

ชุดที่สุ่มจริงได้ Adam 4 configurations และ AdamW 1 configuration ไม่มี SGD ใน 5 ชุดนี้ จึงสรุปความเหนือกว่า/ด้อยกว่าของ SGD จากงานนี้ไม่ได้ ค่า LR/WD ที่ชนะหมายถึงดีที่สุดในชุดที่ทดลอง ไม่ใช่ดีที่สุดใน search space ทั้งหมด

Training max 30 epochs, early stopping patience 7/min_delta .0001 และ ReduceLROnPlateau(mode=max, factor=.5, patience=3) ใช้ validation Macro F1 เป็น objective; checkpoint exact ties ใช้ lower validation loss แล้ว earlier epoch

Tuning ครบ 75 trials ไม่มี failed attempt ใน registry ที่ตรวจ รวม 1,350 training epochs (9–30 epochs/run); final รวม 1,411 epochs (10–30 epochs/run) รวมจริง 2,761 epochs เทียบกับเพดาน 4,500 ไม่รวม checks/smoke หรือการฝึกใน protocol เก่า

![Tuning LR กับ validation](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning/tuning_validation_scatter.png)

![Tuning weight decay กับ validation](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning/weight_decay_validation_scatter.png)

แกน x เป็น log scale, y เป็น best validation F1 ของแต่ละ trial; marker แยก mode จุดรวมหลาย architectures ดังนั้นดูแนวโน้มได้ แต่ไม่ควรตีความว่า LR/WD เพียงตัวเดียวเป็นสาเหตุของคะแนน เพราะ optimizer/batch/architecture เปลี่ยนร่วมด้วย

### สูตรชนะของทั้ง 15 คู่

| Architecture | Mode | Trial | LR | WD | Optimizer | Batch | Val F1 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resnet50 | full | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 93.17 |
| resnet50 | frozen | 3 | 0.0003328736 | 0.001393982 | adam | 16 | 87.76 |
| resnet50 | partial | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 94.40 |
| efficientnet_b0 | full | 2 | 0.0002481624 | 2.380726E-06 | adam | 32 | 94.68 |
| efficientnet_b0 | frozen | 1 | 0.0003531117 | 5.695263E-05 | adam | 32 | 85.04 |
| efficientnet_b0 | partial | 3 | 0.0003328736 | 0.001393982 | adam | 16 | 88.87 |
| mobilenet_v3_large | full | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 94.68 |
| mobilenet_v3_large | frozen | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 92.50 |
| mobilenet_v3_large | partial | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 97.34 |
| densenet121 | full | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 96.28 |
| densenet121 | frozen | 3 | 0.0003328736 | 0.001393982 | adam | 16 | 89.06 |
| densenet121 | partial | 3 | 0.0003328736 | 0.001393982 | adam | 16 | 90.61 |
| convnext_tiny | full | 2 | 0.0002481624 | 2.380726E-06 | adam | 32 | 94.63 |
| convnext_tiny | frozen | 3 | 0.0003328736 | 0.001393982 | adam | 16 | 89.02 |
| convnext_tiny | partial | 5 | 0.0001939681 | 0.001954549 | adam | 16 | 94.90 |

แหล่งข้อมูล: [search definition](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning/search_definition.json), [winners](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning/best_params_by_arch_mode.json), [trial results](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning/all_trials.csv)

## 8. Lock และ final: ต่างจาก tuning อย่างไร

Lock เก็บสูตรชนะทั้ง 15 คู่, reference pair, seeds, preprocessing/loss/mode และ planned statistical families ก่อนอ่าน test ใน `selection.json` เลือก reference ด้วย validation F1 → validation accuracy → lower loss → fixed architecture/mode order ไม่ใช้ test ranking

Reference คือ MobileNetV3-Large/Partial ชนะ trial 5: Adam, LR .0001939681, weight decay .0019545485, batch 16; representative seed = 42

Final เริ่ม **ใหม่จาก pretrained ImageNet weights** ไม่เอา tuning checkpoint มาฝึกต่อ ใช้สูตรชนะของแต่ละคู่กับ seeds `[42,123,456,789,1024]` ฝึกและเลือก checkpoint ด้วย validation เช่นเดิม จากนั้นโหลด checkpoint ที่เลือกเพื่อทำนาย test 85 ภาพครั้งเดียวต่อ run สรุปข้าม seeds โดย join seed ID ตรงกัน

การเลือกดีที่สุดด้วย validation กับการ normalize ภาพเป็นคนละเรื่อง การให้ Full/Frozen/Partial tune ของตัวเองแก้เรื่อง fairness ของ search budget ส่วนภาพทุกคู่ใช้ normalization/split เดียวกัน

## 9. ผล final ครบ 15 คู่

ตารางนี้ทุก metric เป็นเปอร์เซ็นต์ **mean ± sample SD** ของ 5 seeds; Macro metrics ให้น้ำหนักแต่ละ class เท่ากัน

| Architecture | Mode | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- | --- |
| resnet50 | full | 93.65 ± 1.34 | 92.92 ± 2.07 | 92.77 ± 1.69 | 92.56 ± 1.81 |
| resnet50 | frozen | 88.47 ± 1.53 | 87.93 ± 1.72 | 87.65 ± 1.59 | 87.55 ± 1.67 |
| resnet50 | partial | 93.18 ± 1.53 | 92.39 ± 1.63 | 92.11 ± 1.80 | 91.82 ± 2.09 |
| efficientnet_b0 | full | 91.06 ± 1.34 | 89.98 ± 1.43 | 89.25 ± 1.53 | 89.21 ± 1.52 |
| efficientnet_b0 | frozen | 85.65 ± 2.41 | 83.66 ± 2.62 | 83.78 ± 2.30 | 83.38 ± 2.36 |
| efficientnet_b0 | partial | 92.71 ± 2.10 | 91.66 ± 2.64 | 91.29 ± 2.46 | 91.05 ± 2.67 |
| mobilenet_v3_large | full | 91.76 ± 0.83 | 91.45 ± 1.57 | 90.40 ± 0.85 | 89.86 ± 0.96 |
| mobilenet_v3_large | frozen | 90.35 ± 0.98 | 89.14 ± 0.93 | 89.23 ± 1.12 | 88.69 ± 1.33 |
| mobilenet_v3_large | partial | 93.18 ± 2.41 | 92.71 ± 2.04 | 92.17 ± 2.66 | 91.68 ± 3.05 |
| densenet121 | full | 92.71 ± 1.53 | 92.02 ± 1.93 | 91.44 ± 1.85 | 91.27 ± 1.89 |
| densenet121 | frozen | 88.00 ± 1.29 | 86.37 ± 1.73 | 85.86 ± 1.57 | 85.87 ± 1.65 |
| densenet121 | partial | 92.71 ± 1.74 | 91.81 ± 2.10 | 91.49 ± 2.00 | 91.42 ± 1.85 |
| convnext_tiny | full | 92.94 ± 1.66 | 92.36 ± 1.94 | 91.79 ± 1.90 | 91.35 ± 2.12 |
| convnext_tiny | frozen | 87.06 ± 2.76 | 86.14 ± 3.52 | 85.66 ± 2.69 | 85.17 ± 2.48 |
| convnext_tiny | partial | 92.00 ± 1.53 | 92.05 ± 1.46 | 90.80 ± 1.72 | 90.74 ± 1.78 |

![Test Accuracy comparison](C:/Users/User/Desktop/deep/outputs_modes_v3_td/report_assets/pair_accuracy.png)

![Test Macro F1 comparison](C:/Users/User/Desktop/deep/outputs_modes_v3_td/report_assets/pair_macro_f1.png)

กราฟเปรียบเทียบ mean และ sample SD ข้าม 5 seeds อ่านทั้งระดับคะแนนและความแปรปรวน Partial ได้ mean F1 สูงสุดภายใน EfficientNet, MobileNet และ DenseNet; Full ได้สูงสุดภายใน ResNet และ ConvNeXt Frozen ต่ำสุดภายในทุก architecture แต่เรื่อง significance ต้องอ่าน paired tests แยก

ResNet Full เหนือ reference MobileNet Partial ใน mean F1 ประมาณ 0.88 percentage points แต่ planned paired test ของคู่นี้ไม่ significant หลัง Holm (adjusted p=1.0) จึงไม่อ้างว่าชนะอย่างแน่นอน Reference ที่เลือกจาก validation ยังคงเดิม

### ผลราย class ของ reference

| Class | Test support | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- |
| san_phraphrom | 16 | 84.48 ± 5.19 | 100.00 ± 0.00 | 91.52 ± 3.12 |
| san_phraphum | 18 | 95.89 ± 3.77 | 72.22 ± 10.39 | 81.93 ± 6.75 |
| san_tayay | 21 | 90.47 ± 4.05 | 97.14 ± 2.61 | 93.61 ± 1.74 |
| tee_ju_eia | 30 | 100.00 ± 0.00 | 99.33 ± 1.49 | 99.66 ± 0.76 |

San_phraphum เป็น class ที่ reference มี recall/F1 ต่ำสุด: recall เฉลี่ย 72.22%, F1 81.93% เทียบกับ tee_ju_eia F1 99.66% การดู accuracy รวมเพียงอย่างเดียวจะซ่อนปัญหานี้ Recall ของ san_phraphrom 100% ไม่ได้หมายความว่า precision 100% เพราะมี class อื่นถูกทายเป็นพระพรหม

รายละเอียดทุก class ของทั้ง 15 คู่: [per-class summary](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/per_class_summary.csv)

## 10. Training curves และ confusion matrix

![Reference seed42 training curves](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/mobilenet_v3_large/partial/seed42.png)

Loss ดูการลด error ระหว่าง train/validation; Accuracy ดูสัดส่วนถูก; Macro F1 ดูสมดุลข้าม class ทุกเส้นใช้ epoch histories ที่บันทึกไว้ อย่าเลือก epoch ใหม่จาก test Reference seed42 เลือก epoch 10 และหยุดที่ epoch 17; run นี้มี test Accuracy 95.29%, F1 94.27% ซึ่งสูงกว่าค่าเฉลี่ย 5 seeds จึงไม่ใช้ seed นี้แทนผลรวมทั้งหมด

![confusion_matrix.png reference seed42](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/mobilenet_v3_large/partial/final_mobilenet_v3_large_partial_final_seed42_attempt01_seed42_attempt1791931193_1fcd55/confusion_matrix.png)

![confusion_matrix_normalized.png reference seed42](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/mobilenet_v3_large/partial/final_mobilenet_v3_large_partial_final_seed42_attempt01_seed42_attempt1791931193_1fcd55/confusion_matrix_normalized.png)

Confusion matrix จำนวนภาพ: แถว true class คอลัมน์ predicted class; diagonal คือทายถูก ส่วน row-normalized คือสัดส่วนใน true class นั้น ไม่ใช่ precision ของ predicted class ภาพนี้เป็น seed42 เท่านั้น ไม่ใช่ confusion รวม 5 seeds

![Training gaps](C:/Users/User/Desktop/deep/outputs_modes_v3_td/diagnostics/training_gaps.png)

ซ้าย validation−train loss; ขวา train−validation F1 ที่ selected checkpoint แล้วเฉลี่ยข้าม seeds จากภาพ Frozen หลายแบบ gap เล็กกว่าการเปิด backbone แต่คะแนน test ก็ต่ำกว่า จึงตีความว่า gap เล็ก = โมเดลดีกว่าโดยอัตโนมัติไม่ได้ Train มี augmentation ขณะที่ validation ไม่มี จึงเป็น diagnostic ไม่ใช่การวัด overfitting โดยปราศจากปัจจัยอื่น

## 11. Paired statistics: ผลไหนมีนัยสำคัญ

ใช้ two-sided paired t-test คู่ seeds เดียวกัน n=5, alpha .05 และปรับ Holm แยก families ที่ล็อกไว้: primary 28 comparisons (reference เทียบ 14 คู่อื่น × 2 metrics), secondary 30 comparisons (mode pairs ภายใน 5 architectures × 2 metrics) ครบ 58 rows ทุก row มี computational status `ok` แต่ไม่ได้แปลว่า significant ทุก row

พบ significant หลัง Holm **2/28 primary และ 3/30 secondary** ดังตาราง ผลต่างเป็น reference−competitor หน่วย percentage points ค่า negative หมายถึง competitor ดีกว่า

| Family | Reference | Competitor | Metric | Difference (pp) | Holm p |
| --- | --- | --- | --- | --- | --- |
| primary_reference_pair_28 | mobilenet_v3_large::partial | convnext_tiny::frozen | test_acc | 6.12 | 0.01235 |
| primary_reference_pair_28 | mobilenet_v3_large::partial | convnext_tiny::frozen | test_macro_f1 | 6.50 | 0.01235 |
| secondary_within_arch_mode_30 | resnet50::full | resnet50::frozen | test_acc | 5.18 | 0.02777 |
| secondary_within_arch_mode_30 | densenet121::frozen | densenet121::partial | test_acc | -4.71 | 0.02506 |
| secondary_within_arch_mode_30 | densenet121::frozen | densenet121::partial | test_macro_f1 | -5.55 | 0.01971 |

Primary พบ reference ดีกว่า ConvNeXt Frozen ในทั้งสอง metrics เท่านั้น หลังการปรับหลาย comparisons ส่วน secondary พบ ResNet Full ดีกว่า Frozen ใน Accuracy และ DenseNet Partial ดีกว่า Frozen ใน Accuracy/F1 ไม่พบหลักฐานหลัง Holm ที่จะสรุปว่า Full หรือ Partial ชนะกันทุก architecture

คู่ที่ไม่ significant ไม่ได้พิสูจน์ว่าเท่ากัน ผลสะท้อน seed variation บน split เดียว n=5 และไม่ใช่การทดลองกับข้อมูลใหม่ 5 datasets ดู effect size และ SD ประกอบเสมอ [ตารางครบ 58 comparisons](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/statistical_tests.csv)

## 12. Confidence และ calibration

Confidence คือ probability สูงสุดที่โมเดลให้ ไม่ใช่ความถูกต้องที่รับประกัน ECE ใช้ 10 bins เปรียบเทียบ confidence เฉลี่ยกับ accuracy จริงในแต่ละ bin ค่า ECE ต่ำกว่าหมายถึง calibration ดีขึ้นตามนิยามนี้ ไม่ได้หมายความว่า F1 สูงกว่าทุกกรณี

| Architecture | Mode | ECE mean ± sample SD (0–1) |
| --- | --- | --- |
| resnet50 | full | 0.04941 ± 0.01565 |
| resnet50 | frozen | 0.19657 ± 0.01610 |
| resnet50 | partial | 0.05552 ± 0.01466 |
| efficientnet_b0 | full | 0.06964 ± 0.00992 |
| efficientnet_b0 | frozen | 0.15212 ± 0.04564 |
| efficientnet_b0 | partial | 0.06232 ± 0.03235 |
| mobilenet_v3_large | full | 0.07093 ± 0.01896 |
| mobilenet_v3_large | frozen | 0.06948 ± 0.00888 |
| mobilenet_v3_large | partial | 0.04528 ± 0.01666 |
| densenet121 | full | 0.05296 ± 0.01137 |
| densenet121 | frozen | 0.12789 ± 0.01634 |
| densenet121 | partial | 0.04437 ± 0.01184 |
| convnext_tiny | full | 0.05355 ± 0.01648 |
| convnext_tiny | frozen | 0.13867 ± 0.02098 |
| convnext_tiny | partial | 0.04782 ± 0.01138 |

DenseNet Partial มี mean ECE ต่ำสุดในตาราง (.04437); reference อยู่ .04528 ขณะที่ ResNet Frozen สูงสุด .19657 การเรียง ECE เป็น descriptive ranking ไม่ได้มี calibration significance test ใน protocol นี้

![Reference reliability seed42](C:/Users/User/Desktop/deep/outputs_modes_v3_td/confidence/reliability_mobilenet_v3_large_partial_seed42.png)

Reliability ดู bin accuracy เทียบ confidence/เส้น ideal; bin ว่างหรือจำนวนภาพน้อยทำให้ตีความได้จำกัด โดยเฉพาะ test มีเพียง 85 ภาพ

![Reference confidence seed42](C:/Users/User/Desktop/deep/outputs_modes_v3_td/confidence/confidence_hist_mobilenet_v3_large_partial_seed42.png)

Histogram แยกความมั่นใจของภาพถูก/ผิด ถ้าผิดแต่ confidence สูงเป็นกรณีที่ควรตรวจเป็นพิเศษ กราฟตัวอย่างนี้เป็น seed42 แต่ ECE table เป็น 5-seed summary มี reliability/histogram ครบ 15 คู่ที่ seed42 ในโฟลเดอร์ confidence

## 13. Grad-CAM และ eyeball: โมเดลดูส่วนไหนและผิดแบบใด

ใช้ reference ที่ล็อกไว้ MobileNet Partial seed42 ไม่เปลี่ยนไปใช้ ResNet เพราะมี mean test สูงกว่า เลือก 8 ภาพ deterministic: 7 ถูกและ 1 ผิด คลาสที่ไม่มีตัวอย่างผิดใน seed42 ใช้ภาพถูกที่ confidence ต่ำเป็น difficult fallback ใน CSV บางภาพ fallback ถูกระบุ `correct case` จึงไม่ควรอ้างว่าเป็น misclassification

![Grad-CAM reference grid](C:/Users/User/Desktop/deep/outputs_modes_v3_td/gradcam/reference_pair_cam_grid.png)

แต่ละแถวมี original, heatmap และ overlay สีร้อนคือบริเวณที่เกี่ยวข้องกับ score ของ explained target ไม่ใช่ segmentation mask ในแถวผิด target คือ predicted class san_tayay ไม่ใช่ true class san_phraphum

จากการดูภาพ หลายกรณีจุดร้อนอยู่บริเวณโครงสร้างกลางศาล/หลังคา/ฐาน ภาพ `san_phraphum_0076` ทายเป็นศาลตายายด้วย confidence .6001 และ heatmap กระจุกใกล้ตัวศาลด้านล่างขวา แต่ไม่ได้พิสูจน์เชิงเหตุผลว่าการโฟกัสตำแหน่งนั้นทำให้ทายผิด ตัวอย่างบางภาพมีข้อความ/โลโก้ และตัวอย่างศาลตี่จู้มีตัวอักษรอยู่ตรงบริเวณจุดร้อน จึงยังแยกไม่ได้ว่าโมเดลใช้รูปทรงศาลหรือข้อความเพียงอย่างเดียว CAM ไม่ยืนยันว่า shortcut ถูกกำจัดแล้ว

![Eyeball original/ImageNet/fine-tuned](C:/Users/User/Desktop/deep/outputs_modes_v3_td/eyeball/eyeball_grid.png)

Eyeball เปรียบเทียบภาพเดียวกันกับป้าย ImageNet และคำตอบ fine-tuned เช่นภาพพระพรหมได้ ImageNet stupa, พระภูมิได้ altar และศาลตี่จู้บางภาพได้ china cabinet ก่อน fine-tune ข้อมูลทั้งหมดมาจาก prediction logs ไม่แต่งตัวอย่างผิดขึ้นใหม่

| Image ID | True | Predicted | Confidence (%) | ImageNet top1 | Logged reason |
| --- | --- | --- | --- | --- | --- |
| san_phraphrom/san_phraphrom_0057 | san_phraphrom | san_phraphrom | 100.00 | stupa | correct case |
| san_phraphrom/san_phraphrom_0066 | san_phraphrom | san_phraphrom | 61.49 | stupa | correct case |
| san_phraphum/san_phraphum_0093 | san_phraphum | san_phraphum | 100.00 | altar | correct case |
| san_phraphum/san_phraphum_0076 | san_phraphum | san_tayay | 60.01 | altar | misclassified case |
| san_tayay/san_tayay_0068 | san_tayay | san_tayay | 100.00 | boathouse | correct case |
| san_tayay/san_tayay_0003 | san_tayay | san_tayay | 72.19 | church | correct case |
| tee_ju_eia/tee_ju_eia_0001 | tee_ju_eia | tee_ju_eia | 100.00 | china cabinet | correct case |
| tee_ju_eia/tee_ju_eia_0080 | tee_ju_eia | tee_ju_eia | 86.00 | vestment | correct case |

[case selection และ probabilities](C:/Users/User/Desktop/deep/outputs_modes_v3_td/eyeball/eyeball_cases.csv) และ individual original/heatmap/overlay อยู่ใน `gradcam/samples/`

## 14. กราฟที่มีทั้งหมดและวิธีค้น

| โฟลเดอร์ | จำนวน PNG | ตำแหน่ง |
| --- | --- | --- |
| baseline | 1 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/baseline) |
| confidence | 30 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/confidence) |
| dataset_audit | 5 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/dataset_audit) |
| diagnostics | 1 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/diagnostics) |
| eyeball | 1 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/eyeball) |
| final | 240 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final) |
| gradcam | 25 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/gradcam) |
| report_assets | 2 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/report_assets) |
| tuning | 77 | [เปิดโฟลเดอร์](C:/Users/User/Desktop/deep/outputs_modes_v3_td/tuning) |

กราฟ EDA/baseline เป็น copied historical assets ที่ตรวจความเข้ากันได้ ส่วน tuning/final/diagnostics/CAM เป็นผลใน output ล่าสุด ตาราง inventory นับไฟล์ PNG ที่มีจริง ไม่ถือว่าทุกไฟล์เป็นผล training อิสระ กราฟราย run มี identity และ seed ใน directory; กราฟ curves ใน `final/curves` เป็นชุด representative seed42 เพิ่มเติมจากกราฟใน run directories

กราฟ training curves ที่รวบรวมไว้สำหรับ representative seed42 ของทุกคู่:

- [convnext_tiny/frozen/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/convnext_tiny/frozen/seed42.png)
- [convnext_tiny/full/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/convnext_tiny/full/seed42.png)
- [convnext_tiny/partial/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/convnext_tiny/partial/seed42.png)
- [densenet121/frozen/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/densenet121/frozen/seed42.png)
- [densenet121/full/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/densenet121/full/seed42.png)
- [densenet121/partial/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/densenet121/partial/seed42.png)
- [efficientnet_b0/frozen/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/efficientnet_b0/frozen/seed42.png)
- [efficientnet_b0/full/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/efficientnet_b0/full/seed42.png)
- [efficientnet_b0/partial/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/efficientnet_b0/partial/seed42.png)
- [mobilenet_v3_large/frozen/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/mobilenet_v3_large/frozen/seed42.png)
- [mobilenet_v3_large/full/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/mobilenet_v3_large/full/seed42.png)
- [mobilenet_v3_large/partial/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/mobilenet_v3_large/partial/seed42.png)
- [resnet50/frozen/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/resnet50/frozen/seed42.png)
- [resnet50/full/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/resnet50/full/seed42.png)
- [resnet50/partial/seed42.png](C:/Users/User/Desktop/deep/outputs_modes_v3_td/final/curves/resnet50/partial/seed42.png)

## 15. Tracking, runtime และข้อจำกัด

Environment ที่บันทึก: Windows, Python 3.13.5, PyTorch 2.13.0+cu130, torchvision .28.0+cu130, CUDA 13.0, GPU NVIDIA GeForce RTX 5060; reference config ระบุ pretrained=true, AMP=false และ mode modules ชัดเจน

Registry ของ tuning/final ระบุ effective tracking_mode=online ทั้ง 150 runs และเก็บ W&B run URLs ภายใต้ project `thai-shrine-classifier-td` [W&B project ตาม URL ที่บันทึก](https://wandb.ai/wichaidit-m-nida/thai-shrine-classifier-td) อย่างไรก็ตามการอ่าน local registry ไม่ใช่การตรวจ dashboard ว่า artifacts ทุกชิ้นปรากฏบน cloud ครบ ไม่มี cloud Sweep URL เพราะใช้ local tuner

- Test split เดิมเคยถูกดูในงานก่อนหน้า งานนี้จึงเป็น exploratory follow-up บน split เดิม ไม่ใช่การยืนยันกับ external holdout ใหม่
- Search มีเพียง 5 configurations ไม่ครอบคลุม optimizer ทั้งหมด และ tuning ใช้ seed42 เพียง seed เดียวก่อนเลือกสูตร อาจไวต่อ seed; final 5 seeds ช่วยวัดความแปรปรวนภายหลัง
- Mode comparison นี้แต่ละ mode มี hyperparameters ต่างกัน จึงตอบว่า mode ที่ได้รับ tuning ของตัวเองให้ผลอย่างไร ไม่ได้แยก causal effect ของการ freeze เพียงปัจจัยเดียว
- ไม่ได้รัน optional no-augmentation/no-class-weight/padded-input ablation รอบนี้ จึงไม่อ้างผลของการถอด augmentation/weights จากผล mode comparison
- Provenance ไม่ทราบ 112 ภาพและภาพมี backgrounds/text จึงยังมี source/shortcut bias ได้ Neutral canvas และ CAM ไม่ใช่หลักฐานว่ากำจัดหมด
- SD/paired tests ใช้ split เดียวและ 5 seeds; ไม่แทน generalization ไปภาพใหม่ทุกแหล่ง
- Instructor PDF ไม่อยู่ในโฟลเดอร์ จึงยังตรวจข้อกำหนดผู้สอนจากต้นฉบับไม่ได้

ปัญหา implementation ที่พบระหว่างทำ analysis คือ `lock` และ tuning plots เคยอ่าน LR/WD จากด้านนอก record แทน nested `config` แก้ให้อ่านค่าจริงแล้ว ผู้ใช้ rerun lock/analyze จนถึง report โดยไม่ฝึกใหม่เพราะ extraction/plot fix นี้ ตัวเลขสรุปในเอกสารมาจาก logs หลังขั้นตอนเหล่านั้น

## 16. แผนที่ไฟล์สำหรับตรวจย้อนกลับ

| ต้องการดู | ไฟล์/โฟลเดอร์ใน outputs_modes_v3_td |
|---|---|
| protocol/search space/budget | protocol.json, preflight.json, tuning/search_definition.json |
| ข้อมูลและ split | source_manifest.json, splits.csv, split_metadata.json, dataset_audit/ |
| สูตรที่ชนะและ reference lock | tuning/best_params_by_arch_mode.json, selection.json |
| ทุก attempt/สถานะ/tracking URL | run_registry.json |
| ทุก epoch และ selected checkpoints | tuning/ และ final/ ภายในแต่ละ architecture/mode/run |
| 75 final results | final/run_metrics.csv |
| 15 pair mean/SD | final/pair_summary.csv |
| ราย class ของทุกคู่ | final/per_class_summary.csv |
| Paired tests ทั้งหมด | final/statistical_tests.csv |
| Calibration | confidence/run_metrics.csv, pair_summary.csv และ reliability/histogram PNG |
| Training gaps | diagnostics/run_dynamics.csv, training_gaps.png |
| CAM และ selected cases | gradcam/, eyeball/ |
| HTML อธิบายผลเดิม | report_assets/understanding_th.html |

ผลสรุปที่ใช้ตัดสินเชิง protocol คือ validation-selected MobileNet Partial และผล descriptive ที่ mean test สูงสุดคือ ResNet Full ทั้งสองข้อความอยู่ร่วมกันได้: validation ใช้เลือกก่อน test ส่วน test ใช้ประเมินและรายงานภายหลัง ไม่ย้อนกลับไปเปลี่ยน selection
