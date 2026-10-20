# DADS7202 — การจำแนกภาพศาลไทยด้วย Transfer Learning

ผลการทดลองจาก `outputs_modes_v3_td` · ปรับปรุง 20 ตุลาคม 2026

## บทคัดย่อ

งานนี้เปรียบเทียบการทำ transfer learning เพื่อจำแนกภาพศาลไทย 4 คลาส ได้แก่ พระพรหม พระภูมิ ตายาย และตี่จู้เอี๊ยะ จากภาพ 560 ภาพ เราทดลอง 5 โมเดลที่ pretrain บน ImageNet ด้วย 3 mode (Full, Frozen, Partial) รวม 15 setups (โมเดล × mode) แต่ละ setup ค้นหา hyperparameters 5 trials แล้วฝึกซ้ำ 5 seeds รวม 75 trials และ 75 final runs ภาพแบ่งเป็น train 390 / validation 85 / test 85

**ResNet50/Full ได้ Accuracy และ Macro F1 เฉลี่ยบน test สูงสุดจาก 15 setups** โดยได้ Accuracy 93.65% ± 1.34 pp และ Macro F1 92.56% ± 1.81 pp โมเดลอ้างอิง MobileNetV3-Large/Partial (เลือกจาก validation ก่อนดู test) ได้ 93.18% ± 2.41 pp และ 91.68% ± 3.05 pp ความต่างนี้ยังสรุปไม่ได้ทางสถิติหลังปรับ Holm คลาสพระภูมิยากที่สุดสำหรับทั้งสองโมเดล โดย ResNet50/Full ได้ Recall 77.78% และ MobileNetV3-Large/Partial ได้ 72.22%

เราทำ ablation กับ ResNet50/Full เพิ่ม 10 runs (รวมทั้งงาน 160 runs) เมื่อปิด augmentation พบว่า Macro F1 เฉลี่ยลด 1.32 pp และเมื่อปิด class weights ลด 0.67 pp ทั้งสองกรณียังไม่พบความแตกต่างอย่างมีนัยสำคัญหลังปรับ Holm นอกจากนี้เราสร้าง Grad-CAM จาก checkpoint เดิมโดยไม่ฝึกเพิ่ม

เราเลือก ResNet50/Full หลังดูผล test และ test มีเพียง 85 ภาพ ข้อสรุปนี้จึงเป็นผลเชิงสำรวจ ต้องประเมินยืนยันด้วยภาพจากแหล่งภายนอกชุดใหม่

---

## 1) Dataset description

### 1.1 โจทย์และแหล่งข้อมูล

โจทย์คือจำแนกภาพศาล 4 คลาสที่มีรูปทรงและบริบทแตกต่างกัน ข้อมูลอยู่ใน `dataset_cleaned/` และผ่านการตรวจป้ายกำกับโดยมนุษย์แล้ว ตารางใช้รหัสคลาสเดียวกับ logits และไฟล์ผลทุกไฟล์

| คลาส | Class ID | จำนวนภาพ | สัดส่วน (%) |
| --- | --- | ---: | ---: |
| พระพรหม | san_phraphrom | 103 | 18.39 |
| พระภูมิ | san_phraphum | 124 | 22.14 |
| ตายาย | san_tayay | 133 | 23.75 |
| ตี่จู้เอี๊ยะ | tee_ju_eia | 200 | 35.71 |

ไฟล์ `metadata_cleaned.csv` เก็บชื่อไฟล์ คลาส และ URL ต้นทาง มี URL ระบุไว้ 448 ภาพ และไม่ทราบ URL 112 ภาพ โดเมนที่พบมากที่สุดคือ `down-th.img.susercontent.com` (65 ภาพ) รายการเต็มอยู่ใน Appendix

### 1.2 การสำรวจข้อมูล

ตี่จู้เอี๊ยะมีภาพมากที่สุด (200 ภาพ) และพระพรหมน้อยที่สุด (103 ภาพ) อัตราส่วนประมาณ 1.94 : 1 เพราะ class imbalance นี้ เราจึงใช้ทั้ง Macro F1 และ class-weighted loss

![ตัวอย่างภาพทั้งสี่คลาส](report/report_assets/sample_grid.png)

*ภาพที่ 1 ตัวอย่างภาพแต่ละคลาส แสดงความต่างของรูปทรง มุมมอง และฉากหลัง*

![การกระจายจำนวนภาพ](report/report_assets/latest/class_distribution.svg)

*ภาพที่ 2 จำนวนภาพต่อคลาส*

![ขนาดภาพ](report/report_assets/latest/image_dimensions.svg)

*ภาพที่ 3 ขนาดและอัตราส่วนกว้างต่อสูงของภาพต้นฉบับ 560 ภาพ เส้นประคือภาพจัตุรัส*

![โหมดสีของภาพ](report/report_assets/latest/colour_modes.svg)

*ภาพที่ 4 โหมดสีของไฟล์ภาพ ภาพส่วนน้อยต้องจัดการ alpha, palette หรือ CMYK ก่อนเข้าโมเดล*

![ความสว่างของภาพ](report/report_assets/latest/brightness.svg)

*ภาพที่ 5 ความสว่างเฉลี่ย (0–255) ของภาพต้นฉบับแยกตามคลาส*

<sub>Note ภาพที่ 5: คำนวณหลังหมุนตาม EXIF และผสม alpha กับพื้นกลาง ใช้ Rec.709 luminance กล่องคือ IQR และมัธยฐาน เส้นปลายคือค่าต่ำสุด–สูงสุด</sub>

ภาพมีขนาดต่างกัน 206 แบบ ความกว้าง 145–4,032 พิกเซล ความสูง 225–4,214 พิกเซล โหมดสีเป็น RGB 543 ภาพ, RGBA 9 ภาพ, CMYK 3 ภาพ และ palette 5 ภาพ ส่วนไฟล์เป็น JPEG 526 ภาพ, PNG 22 ภาพ, WEBP 11 ภาพ และ GIF 1 ภาพ เราจึงต้องแปลงทุกภาพให้เป็นรูปแบบเดียวกันก่อนเข้าโมเดล

### 1.3 ภาพซ้ำและความสมบูรณ์ของข้อมูล

ไม่พบไฟล์ที่อ่านไม่ได้ พบภาพซ้ำแบบ decoded RGB ในคลาสพระภูมิ 2 คู่ (`0002/0046` และ `0073/0077`) เมื่อรวมกับกลุ่มภาพใกล้เคียงที่ตรวจด้วยมือ ได้ 550 กลุ่ม เราแบ่งข้อมูลโดยใช้กลุ่มเป็นหน่วย ภาพที่เกี่ยวข้องกันจึงอยู่ใน split เดียวกัน

---

## 2) Data preparation

### 2.1 การแบ่ง train, validation และ test

เราใช้ split seed 42 และเลือกผลจาก GroupShuffleSplit แบบ deterministic 256 × 12 candidates ที่ใกล้สัดส่วน 70/15/15 และรักษาสัดส่วนคลาส ทุกโมเดล ทุก mode และทุก seed ใช้ภาพชุดเดียวกัน

| Split | รวม | พระพรหม | พระภูมิ | ตายาย | ตี่จู้เอี๊ยะ |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 390 | 70 | 89 | 91 | 140 |
| validation | 85 | 17 | 17 | 21 | 30 |
| test | 85 | 16 | 18 | 21 | 30 |

จาก `splits.csv` ไม่พบกลุ่มภาพที่ข้าม split และทุก split มีครบสี่คลาส Train ใช้ปรับน้ำหนัก validation ใช้เลือก hyperparameters และ checkpoint ส่วน test ใช้ประเมินหลังล็อกการทดลอง

### 2.2 การเตรียมภาพและ augmentation

ทุกภาพผ่านขั้นตอนนี้ตามลำดับ

1. ถอดรหัสภาพ หมุนตาม EXIF แปลงเป็น RGB (ส่วนโปร่งใสผสมกับพื้นกลาง)
2. Resize ด้วย LANCZOS โดยรักษาอัตราส่วน ให้ด้านยาวเท่ากับ 224 พิกเซล
3. วางภาพกึ่งกลางบนพื้น RGB `(124,116,104)` ขนาด 224 × 224

วิธีนี้รักษาสัดส่วนของศาลไว้

![ตัวอย่าง padding](report/report_assets/latest/padding_large.png)

*ภาพที่ 6 ภาพ train `san_phraphrom_0001` ขนาด 960 × 720 เทียบกับผล preprocessing 224 × 224 ที่ขยายให้เห็นรายละเอียด*

พื้นที่เพิ่มใช้ RGB (124,116,104) และเก็บขอบเดิมที่ติดมากับภาพต้นฉบับไว้

Augmentation ใช้กับ train เท่านั้น

| การแปลง | ค่าที่ใช้ |
|---|---|
| Rotation | ±12° |
| Translation | ±5% |
| Scale | 0.92–1.08 |
| Perspective | distortion 0.10, probability 0.20 |
| Horizontal flip | probability 0.50 |
| Color jitter | brightness/contrast/saturation 0.10, hue 0.02 |
| Random Erasing | probability 0.05, area 0.02–0.08 |

พื้นที่ว่างจากการแปลงเชิงเรขาคณิตและ erasing เติมด้วยสีพื้นกลางเดียวกัน จากนั้นแปลงเป็น tensor และ normalize ด้วย ImageNet mean `(0.485, 0.456, 0.406)` และ SD `(0.229, 0.224, 0.225)` Validation และ test ใช้ preprocessing แบบคงที่

![ตัวอย่าง augmentation](report/report_assets/latest/augmentation_train_5_per_class.png)

*ภาพที่ 7 ตัวอย่าง augmentation จากภาพ train คลาสละ 5 ภาพ*

<sub>Note ภาพที่ 7: ภาพสร้างใหม่ด้วย seed คงที่เพื่ออธิบายนโยบาย augmentation ไม่ใช่ minibatch ที่บันทึกระหว่างฝึก การเรนเดอร์ด้วย JavaScript อาจต่างจาก torchvision ในระดับพิกเซล ดู [image IDs และค่าการแปลง](report/preprocessing_preview_sources.json)</sub>

DataLoader สุ่มลำดับเฉพาะ train ใช้ `num_workers=0` และ `drop_last=False` เพื่อใช้ภาพครบทุก batch Class weights คำนวณจาก `390 / (4 × จำนวนภาพในคลาส)` ได้ `[1.392857, 1.095506, 1.071429, 0.696429]` ตามลำดับคลาสในตารางข้อ 1.1

---

## 3) Model architecture

### 3.1 โมเดลและ classifier

เราใช้ pretrained weights อย่างเป็นทางการของ torchvision (`Weights.DEFAULT`) และเปลี่ยนเฉพาะ linear layer สุดท้ายให้ส่ง 4 logits ส่วนอื่นของ classifier คงเดิม

| โมเดล | Backbone | สิ่งที่เปลี่ยนใน classifier |
|---|---|---|
| ResNet50 | Residual blocks | `fc` เป็น linear 4 คลาส |
| EfficientNet-B0 | MBConv blocks | linear สุดท้าย (คง Dropout 0.2) |
| MobileNetV3-Large | InvertedResidual blocks | linear สุดท้าย (คง hidden linear, Hardswish, Dropout 0.2) |
| DenseNet121 | Dense blocks | `classifier` เป็น linear 4 คลาส |
| ConvNeXt-Tiny | ConvNeXt stages | linear สุดท้าย (คง normalization และ flatten) |

จำนวน parameters ของแต่ละโมเดลและแต่ละ mode อยู่ใน Appendix (ResNet50 มี 23.52M, EfficientNet-B0 4.01M, MobileNetV3-Large 4.21M, DenseNet121 6.96M, ConvNeXt-Tiny 27.82M)

### 3.2 Mode ของการปรับน้ำหนัก

เรากำหนดส่วนที่ฝึกก่อนสร้าง optimizer ในทุก run

| Mode | ส่วนที่ฝึก | สถานะระหว่างฝึก |
|---|---|---|
| Full | Backbone และ classifier ทั้งหมด | train mode ปกติ |
| Frozen | Classifier head | Backbone อยู่ใน eval mode รวม BatchNorm buffers |
| Partial | Backbone stage สุดท้ายและ classifier | ส่วนต้นอยู่ใน eval mode ส่วนที่เปิดฝึกอยู่ใน train mode |

กราฟเปรียบเทียบใช้ Frozen = น้ำเงิน ●, Partial = ส้มแดง ■ และ Full = เขียว ◆ เหมือนกันตลอดรายงาน ส่วนกราฟข้อมูลใช้สีแยกตามคลาสและคงลำดับคลาสตามตารางข้อ 1.1

Stage ที่เปิดฝึกใน Partial: ResNet50 ใช้ `layer4`, DenseNet121 ใช้ `denseblock4` และ `norm5`, EfficientNet-B0 ใช้ MBConv stage สุดท้ายและ projection, ConvNeXt-Tiny ใช้ feature stage สุดท้าย, MobileNetV3-Large ใช้ `features.13–16` และ `classifier`

ทุก run เริ่มจาก ImageNet weights ใหม่ ดังนั้นการเปรียบเทียบ 3 mode จึงวัดขอบเขตการปรับน้ำหนัก โดยแต่ละ mode ค้นหา hyperparameters แยกกัน

### 3.3 ผลของ ImageNet model ก่อนปรับ classifier

ใช้ผลที่บันทึกไว้ของ ResNet50 ที่ยังมี classifier 1,000 คลาสบนภาพ train เพื่อดูว่าโมเดลเดิมมองภาพศาลอย่างไร ผลคือโมเดลทายศาลเป็นวัตถุอื่น เช่น `stupa` ซึ่งใกล้เคียงรูปทรง แต่แยกศาลไทยสี่คลาสไม่ได้ ตัวเลขมาจาก baseline เดิมที่ใช้ข้อมูลชุดเดียวกัน ภาพประกอบจัดวางใหม่จากตัวอย่าง train เดิม และแสดง confidence เป็นเปอร์เซ็นต์

![ผล ImageNet baseline](report/report_assets/latest/imagenet_baseline_percent.png)

*ภาพที่ 8 ป้ายและ confidence ของ ResNet50/ImageNet ก่อนปรับ classifier ใช้อธิบาย transfer learning เท่านั้น ไม่ได้นำไปคำนวณ Accuracy*

---

## 4) Training method

### 4.1 การค้นหา hyperparameters

ลำดับการทดลอง: เตรียมข้อมูล → ค้นหา hyperparameters ของ 15 setups → เลือก trial ชนะของแต่ละ setup → ล็อกสูตรและโมเดลอ้างอิง → ฝึกซ้ำ 5 seeds → ประเมิน test และวิเคราะห์ผล

เราใช้ local seeded random search setup ละ 5 trials รวม 75 trials และบันทึกแต่ละ trial เป็น W&B run คำว่า sweep ในงานนี้หมายถึงการค้นหาแบบ local random search ที่บันทึกผลผ่าน W&B ขอบเขตการค้นหา:

| Hyperparameter | ช่วง |
| --- | --- |
| Learning rate | log-uniform [1e-5, 1e-3] |
| Weight decay | log-uniform [1e-6, 1e-2] |
| Optimizer | AdamW, Adam, SGD |
| Batch size | 16, 32 |

ทั้ง 15 setups ใช้ชุดค่าสุ่มเดียวกัน แต่เลือก trial ชนะจาก validation ของ setup นั้น ๆ

<!-- generated:SEARCH_TABLE -->

| Trial | Learning rate | Weight decay | Optimizer | Batch |
| ---: | ---: | ---: | --- | ---: |
| 1 | 3.531117e-4 | 5.695263e-5 | Adam | 32 |
| 2 | 2.481624e-4 | 2.380726e-6 | Adam | 32 |
| 3 | 3.328736e-4 | 1.393982e-3 | Adam | 16 |
| 4 | 7.957413e-5 | 3.042230e-5 | AdamW | 32 |
| 5 | 1.939681e-4 | 1.954549e-3 | Adam | 16 |

<!-- /generated:SEARCH_TABLE -->

ชุดที่สุ่มได้มี Adam 4 ชุดและ AdamW 1 ชุด จึงไม่มีผลของ SGD ในการทดลองนี้ การสุ่มเพียง 5 ชุดช่วยคุมงบ แต่ครอบคลุม search space ได้บางส่วน

![ความสัมพันธ์ของ hyperparameters กับ validation Macro F1](report/report_assets/latest/tuning_response.svg)

*ภาพที่ 9 Validation Macro F1 เทียบกับ learning rate และ weight decay จาก 75 trials แกน x เป็น log scale*

<sub>Note ภาพที่ 9: แนวโน้มที่เห็นปนกับผลของโมเดล optimizer และ batch size ที่ต่างกัน วงแหวนระบุ trial ที่ชนะของแต่ละ setup จุดอาจซ้อนกันเพราะใช้ชุดค่าสุ่มเดียวกัน และรอบนี้ไม่ได้ sweep dropout</sub>

### 4.2 เงื่อนไขการฝึกและการเลือก checkpoint

| รายการ | ค่า |
|---|---|
| Loss | Class-weighted cross entropy |
| Epochs สูงสุด | 30 |
| Early stopping | patience 7, `min_delta=0.0001` |
| Scheduler | `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` |
| ค่าที่ scheduler และ early stopping ดู | Validation Macro F1 |
| Adam defaults | betas (0.9, 0.999), epsilon 1e-8 |
| Tuning seed | 42 |
| Final seeds | 42, 123, 456, 789, 1024 |

Checkpoint ที่เลือกคือ epoch ที่ validation Macro F1 สูงสุด ถ้าเท่ากันให้ดู validation loss ที่ต่ำกว่า และถ้ายังเท่ากันให้เลือก epoch ก่อนหน้า สำหรับ trial ชนะ เรียงเกณฑ์ตาม validation Macro F1, validation Accuracy, validation loss และ trial index

### 4.3 สูตรที่เลือกและการประเมินรอบสุดท้าย

ตารางนี้ระบุ trial ที่ชนะของแต่ละ setup ค่า hyperparameters อ่านจาก trial เดียวกันในตารางข้อ 4.1 ส่วน validation Macro F1 มาจาก tuning seed 42

<!-- generated:WINNER_TABLE -->

| Setup | Trial ที่เลือก | Validation Macro F1 (%) |
| --- | ---: | ---: |
| ResNet50/Full | 5 | 93.17 |
| ResNet50/Frozen | 3 | 87.76 |
| ResNet50/Partial | 5 | 94.40 |
| EfficientNet-B0/Full | 2 | 94.68 |
| EfficientNet-B0/Frozen | 1 | 85.04 |
| EfficientNet-B0/Partial | 3 | 88.87 |
| MobileNetV3-Large/Full | 5 | 94.68 |
| MobileNetV3-Large/Frozen | 5 | 92.50 |
| MobileNetV3-Large/Partial | 5 | 97.34 |
| DenseNet121/Full | 5 | 96.28 |
| DenseNet121/Frozen | 3 | 89.06 |
| DenseNet121/Partial | 3 | 90.61 |
| ConvNeXt-Tiny/Full | 2 | 94.63 |
| ConvNeXt-Tiny/Frozen | 3 | 89.02 |
| ConvNeXt-Tiny/Partial | 5 | 94.90 |

<!-- /generated:WINNER_TABLE -->


เราเลือก MobileNetV3-Large/Partial เป็น **โมเดลอ้างอิง** เพราะได้ validation Macro F1 สูงสุด (97.34%) พร้อม Accuracy 97.65% และ loss 0.1718 ผลการเลือกและสูตรทั้ง 15 setups ถูกล็อกใน `selection.json` ก่อนประเมิน test

รอบ final ฝึกใหม่จาก ImageNet weights setup ละ 5 seeds จากนั้นโหลด checkpoint ที่เลือกด้วย validation กลับมาประเมิน test หนึ่งครั้งต่อ run รวม 75 final runs ใช้ 1,411 epochs บวก tuning 1,350 epochs เป็น 2,761 epochs จากเพดาน 4,500 epochs ส่วน ablation 10 runs ใช้ 145 epochs จากเพดาน 300 epochs รวมทั้งงาน **160 runs และ 2,906 epochs**

สภาพแวดล้อม: Windows, Python 3.13.5, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA 13.0, RTX 5060 ปิด AMP และตั้ง deterministic algorithms แบบ `warn_only` ผลจึงอาจต่างเมื่อเปลี่ยน hardware หรือ library version

ไฟล์ประกอบ: [W&B: thai-shrine-classifier-td](https://wandb.ai/wichaidit-m-nida/thai-shrine-classifier-td) · รายละเอียดเรื่อง CoreWeave และสถานะ cloud artifacts อยู่ใน Appendix

---

## 5) Evaluation metrics

ตารางผลใช้ค่าเฉลี่ย (%) ± sample SD (pp) จาก 5 seeds โดย pp หมายถึงจุดเปอร์เซ็นต์ ผลต่างและช่วงความเชื่อมั่นของ Accuracy/F1 ใช้ pp ส่วนค่า p, t และ effect size ไม่มีหน่วยเปอร์เซ็นต์ ไฟล์ CSV/JSON ต้นทางคงหน่วยเดิม

เราวัดด้วย **Accuracy** (สัดส่วนภาพที่ทายถูก) และ **Precision, Recall, F1 รายคลาส**

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2PR / (P + R)
- ถ้าตัวหารเป็นศูนย์ ให้ค่าเป็น 0

**Macro precision, recall และ F1** คือค่าเฉลี่ยของสี่คลาสโดยให้น้ำหนักเท่ากัน คลาสที่มีภาพน้อยจึงมีผลเท่าคลาสที่มีภาพมาก ผลรวมรายงานเป็น mean ± sample SD จาก 5 seeds (`ddof=1`) Confusion matrix ให้ true class เป็นแถว และ predicted class เป็นคอลัมน์

**การทดสอบทางสถิติ:** เราใช้ two-sided paired t-test จับคู่ด้วย seed เดียวกัน เทียบ Accuracy และ Macro F1 แล้วปรับค่า p ด้วย Holm ที่ระดับ 0.05 แยกสองกลุ่มสมมติฐานที่ล็อกไว้

| กลุ่ม | การเปรียบเทียบ | จำนวนสมมติฐาน |
|---|---|---|
| Primary | โมเดลอ้างอิงกับอีก 14 setups × 2 metrics | 28 |
| Secondary | สามคู่ mode ภายในแต่ละโมเดล × 5 โมเดล × 2 metrics | 30 |

เรารายงาน Cohen's d_z (mean ของผลต่างจับคู่ หารด้วย SD ของผลต่าง) และ 95% CI ของ mean difference จาก t-distribution (df = 4) CI นี้ไม่ได้ปรับ multiple comparisons ดังนั้นการตัดสินใช้ Holm-adjusted p

เราวัด calibration ด้วย Expected Calibration Error (ECE) 10 bins ค่า ECE ต่ำหมายถึง confidence ใกล้เคียง accuracy ในแต่ละ bin

---

## 6) Experimental results

### 6.1 ความครบถ้วนของผล

ผลครบ 15 setups × 5 seeds เราตรวจ `test_predictions.csv` ของทั้ง 75 runs (6,375 แถว) พบ test IDs 85 ภาพเรียงตรงกันทุก run และคำนวณ Accuracy, Macro metrics, ผลรายคลาส และ paired tests ซ้ำได้ตรงกับผลที่บันทึก ส่วน ablation ตรวจ 1,275 แถวจาก 15 runs ได้ตรงเช่นกัน ภาพ test ยังเป็น 85 ภาพเดิม จำนวน runs ที่เพิ่มขึ้นจึงไม่ได้เพิ่มตัวอย่างอิสระ

ไฟล์ประกอบ: [verification.json](report/verification.json) · [resnet_followup_verification.json](report/resnet_followup_verification.json)

### 6.2 ผลระหว่างฝึก ณ checkpoint ที่เลือก

กราฟแสดง validation Macro F1 และ best epoch ของ final runs ทั้ง 5 seeds ต่อ mode กล่องคือ IQR และมัธยฐาน ปลายเส้นคือค่าต่ำสุด–สูงสุดที่สังเกต จุดแสดงแต่ละ seed ช่วงแกนตรงกันทุกโมเดล: F1 75–100 และ epoch 0–30

![ResNet50 checkpoint box plots](report/report_assets/latest/checkpoint_boxes_resnet50.svg)

*ภาพที่ 10 Validation Macro F1 (%) และ best epoch ของ ResNet50*

![EfficientNet-B0 checkpoint box plots](report/report_assets/latest/checkpoint_boxes_efficientnet_b0.svg)

*ภาพที่ 11 Validation Macro F1 (%) และ best epoch ของ EfficientNet-B0*

![MobileNetV3-Large checkpoint box plots](report/report_assets/latest/checkpoint_boxes_mobilenet_v3_large.svg)

*ภาพที่ 12 Validation Macro F1 (%) และ best epoch ของ MobileNetV3-Large*

![DenseNet121 checkpoint box plots](report/report_assets/latest/checkpoint_boxes_densenet121.svg)

*ภาพที่ 13 Validation Macro F1 (%) และ best epoch ของ DenseNet121*

![ConvNeXt-Tiny checkpoint box plots](report/report_assets/latest/checkpoint_boxes_convnext_tiny.svg)

*ภาพที่ 14 Validation Macro F1 (%) และ best epoch ของ ConvNeXt-Tiny*

Box plots สรุปค่าที่ checkpoint ส่วนการเปลี่ยนแปลงตามเวลายังดูได้จาก [learning curves เดิมทั้ง 15 setups](report/report_assets/learning_curves/all_pairs_grid.png) ช่วงท้ายอาจเหลือน้อยกว่า 5 runs เพราะ early stopping ดู [จำนวน runs ต่อ epoch](report/learning_curve_epoch_counts.csv) และ [ที่มา/SHA256](report/learning_curve_sources.json)

ตาราง best/stop epochs, Train/Validation F1 และจำนวน runs ที่ loss แยกตัวช่วงท้ายอยู่ใน Appendix C

### 6.3 ผลบน test

**ResNet50/Full ได้ Macro F1 เฉลี่ยสูงสุดจาก 15 setups** (92.56%) ตามด้วย ResNet50/Partial (91.82%) และ MobileNetV3-Large/Partial (91.68%) ทุกค่าเป็น mean (%) ± sample SD (pp) จาก 5 seeds ตารางเรียงตาม Macro F1

<!-- generated:RESULT_TABLE -->

| Setup | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| ResNet50/Full | 93.65 ± 1.34 | 92.92 ± 2.07 | 92.77 ± 1.69 | 92.56 ± 1.81 |
| ResNet50/Partial | 93.18 ± 1.53 | 92.39 ± 1.63 | 92.11 ± 1.80 | 91.82 ± 2.09 |
| MobileNetV3-Large/Partial | 93.18 ± 2.41 | 92.71 ± 2.04 | 92.17 ± 2.66 | 91.68 ± 3.05 |
| DenseNet121/Partial | 92.71 ± 1.74 | 91.81 ± 2.10 | 91.49 ± 2.00 | 91.42 ± 1.85 |
| ConvNeXt-Tiny/Full | 92.94 ± 1.66 | 92.36 ± 1.94 | 91.79 ± 1.90 | 91.35 ± 2.12 |
| DenseNet121/Full | 92.71 ± 1.53 | 92.02 ± 1.93 | 91.44 ± 1.85 | 91.27 ± 1.89 |
| EfficientNet-B0/Partial | 92.71 ± 2.10 | 91.66 ± 2.64 | 91.29 ± 2.46 | 91.05 ± 2.67 |
| ConvNeXt-Tiny/Partial | 92.00 ± 1.53 | 92.05 ± 1.46 | 90.80 ± 1.72 | 90.74 ± 1.78 |
| MobileNetV3-Large/Full | 91.76 ± 0.83 | 91.45 ± 1.57 | 90.40 ± 0.85 | 89.86 ± 0.96 |
| EfficientNet-B0/Full | 91.06 ± 1.34 | 89.98 ± 1.43 | 89.25 ± 1.53 | 89.21 ± 1.52 |
| MobileNetV3-Large/Frozen | 90.35 ± 0.98 | 89.14 ± 0.93 | 89.23 ± 1.12 | 88.69 ± 1.33 |
| ResNet50/Frozen | 88.47 ± 1.53 | 87.93 ± 1.72 | 87.65 ± 1.59 | 87.55 ± 1.67 |
| DenseNet121/Frozen | 88.00 ± 1.29 | 86.37 ± 1.73 | 85.86 ± 1.57 | 85.87 ± 1.65 |
| ConvNeXt-Tiny/Frozen | 87.06 ± 2.76 | 86.14 ± 3.52 | 85.66 ± 2.69 | 85.17 ± 2.48 |
| EfficientNet-B0/Frozen | 85.65 ± 2.41 | 83.66 ± 2.62 | 83.78 ± 2.30 | 83.38 ± 2.36 |

<!-- /generated:RESULT_TABLE -->

![เปรียบเทียบ validation และ test ของ 15 setups](report/report_assets/latest/mode_comparison.svg)

*ภาพที่ 15 Validation และ test Accuracy / Macro F1 ของ 15 setups จุดใหญ่คือค่าเฉลี่ย จุดจางคือแต่ละ seed เส้นคือ ± 1 SD (n = 5)*

ข้อสังเกตจากตาราง

- ผลต่าง Macro F1 ระหว่าง ResNet50/Full กับโมเดลอ้างอิงคือ 0.88 pp
- MobileNetV3-Large/Full มี SD ของ Macro F1 ต่ำที่สุด (0.96 pp) แต่ค่าเฉลี่ยอยู่อันดับ 9 จึงต้องดูคะแนนและความสม่ำเสมอควบคู่กัน
- ภายในโมเดล Partial ให้ Macro F1 เฉลี่ยสูงสุดใน EfficientNet-B0, MobileNetV3-Large และ DenseNet121 ส่วน Full สูงสุดใน ResNet50 และ ConvNeXt-Tiny
- Frozen ต่ำสุดในทั้ง 5 โมเดล

อันดับนี้เป็นผลที่สังเกตหลังล็อกโมเดลอ้างอิงแล้ว ความต่างของค่าเฉลี่ยต้องอ่านร่วมกับผลทางสถิติในหัวข้อ 6.4

### 6.4 ผลการทดสอบทางสถิติ

หลังปรับ Holm มี 2 จาก 28 สมมติฐานใน Primary และ 3 จาก 30 ใน Secondary ที่มีนัยสำคัญ ตารางแสดงรายการเหล่านั้นและการเทียบโมเดลอ้างอิงกับ ResNet50/Full Δ = A − B หน่วย pp

<!-- generated:STATS_TABLE -->

| Family | A − B | Metric | Mean Δ (pp) | 95% CI (pp)¹ | Holm p |
| --- | --- | --- | ---: | --- | ---: |
| Primary | MobileNetV3-Large/Partial<br>− ResNet50/Full | Accuracy | -0.47 | [-4.42, 3.48] | 1.000 |
| Primary | MobileNetV3-Large/Partial<br>− ResNet50/Full | Macro F1 | -0.88 | [-5.89, 4.13] | 1.000 |
| Primary | MobileNetV3-Large/Partial<br>− ConvNeXt-Tiny/Frozen | Accuracy | 6.12 | [4.52, 7.72] | 0.012 |
| Primary | MobileNetV3-Large/Partial<br>− ConvNeXt-Tiny/Frozen | Macro F1 | 6.50 | [4.81, 8.20] | 0.012 |
| Secondary | ResNet50/Full<br>− ResNet50/Frozen | Accuracy | 5.18 | [3.51, 6.84] | 0.028 |
| Secondary | DenseNet121/Frozen<br>− DenseNet121/Partial | Accuracy | -4.71 | [-6.17, -3.25] | 0.025 |
| Secondary | DenseNet121/Frozen<br>− DenseNet121/Partial | Macro F1 | -5.55 | [-7.16, -3.95] | 0.020 |

<!-- /generated:STATS_TABLE -->

ผลที่มีนัยสำคัญ

- โมเดลอ้างอิงสูงกว่า ConvNeXt-Tiny/Frozen ทั้ง Accuracy และ Macro F1
- ResNet50/Full สูงกว่า ResNet50/Frozen ใน Accuracy
- DenseNet121/Partial สูงกว่า DenseNet121/Frozen ทั้ง Accuracy และ Macro F1

ResNet50/Full เทียบกับโมเดลอ้างอิงได้ Holm-adjusted p = 1.000 ทั้งสอง metrics จึงยังไม่มีหลักฐานเพียงพอว่าประสิทธิภาพต่างกัน และไม่ได้ยืนยันว่าทั้งสองโมเดลเท่ากัน

¹ ช่วงความเชื่อมั่น 95% ยังไม่ปรับ multiple comparisons การตัดสินนัยสำคัญใช้ Holm p ส่วนค่า t, raw p และ Cohen’s d_z อยู่ในไฟล์ประกอบ

ไฟล์ประกอบ: [verified_statistics.csv](report/verified_statistics.csv) (58 แถว พร้อม raw p, effect size และ CI)

### 6.5 ผลรายคลาส

เราเทียบ ResNet50/Full (โมเดลที่เลือก) กับ MobileNetV3-Large/Partial (โมเดลอ้างอิง) ค่าเป็น mean (%) ± sample SD (pp) จาก 5 seeds Support คือจำนวนภาพ test ต่อ run

<!-- generated:PERCLASS_TABLE -->

| Setup | คลาส | Support | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50/Full | พระพรหม | 16 | 88.98 ± 6.48 | 97.50 ± 3.42 | 92.94 ± 3.96 |
| ResNet50/Full | พระภูมิ | 18 | 92.17 ± 5.39 | 77.78 ± 3.93 | 84.35 ± 4.43 |
| ResNet50/Full | ตายาย | 21 | 90.55 ± 5.01 | 97.14 ± 2.61 | 93.63 ± 2.31 |
| ResNet50/Full | ตี่จู้เอี๊ยะ | 30 | 100.00 ± 0.00 | 98.67 ± 1.83 | 99.32 ± 0.93 |
| MobileNetV3-Large/Partial | พระพรหม | 16 | 84.48 ± 5.19 | 100.00 ± 0.00 | 91.52 ± 3.12 |
| MobileNetV3-Large/Partial | พระภูมิ | 18 | 95.89 ± 3.77 | 72.22 ± 10.39 | 81.93 ± 6.75 |
| MobileNetV3-Large/Partial | ตายาย | 21 | 90.47 ± 4.05 | 97.14 ± 2.61 | 93.61 ± 1.74 |
| MobileNetV3-Large/Partial | ตี่จู้เอี๊ยะ | 30 | 100.00 ± 0.00 | 99.33 ± 1.49 | 99.66 ± 0.76 |

<!-- /generated:PERCLASS_TABLE -->

พระภูมิเป็นคลาสที่ยากที่สุดของทั้งสองโมเดล ResNet50/Full ได้ Recall 77.78% และ F1 84.35% สูงกว่า MobileNetV3-Large/Partial (72.22% และ 81.93%) แต่ MobileNetV3-Large/Partial ได้ Recall ของพระพรหมและตี่จู้เอี๊ยะสูงกว่า ดังนั้น ResNet50/Full ไม่ได้ชนะทุกคลาส

ไฟล์ประกอบ: [verified_per_class_summary.csv](report/verified_per_class_summary.csv) (ครบ 15 setups)

### 6.6 การเลือกโมเดลสุดท้าย

เราเลือก **ResNet50/Full** โดยดู 4 ด้านพร้อมกัน: ความถูกต้องรวม ความสมดุลระหว่างคลาส ความสม่ำเสมอข้าม seeds และ calibration เราไม่ได้รวมเป็นคะแนนเดียว เพราะ metrics เหล่านี้สัมพันธ์กันอยู่แล้ว

| ค่าวัด (%) | ResNet50/Full (เลือก) | MobileNetV3-Large/Partial (อ้างอิง) |
|---|---|---|
| Accuracy ↑ | 93.65 ± 1.34 | 93.18 ± 2.41 |
| Macro precision ↑ | 92.92 ± 2.07 | 92.71 ± 2.04 |
| Macro recall ↑ | 92.77 ± 1.69 | 92.17 ± 2.66 |
| Macro F1 ↑ | 92.56 ± 1.81 | 91.68 ± 3.05 |
| Recall พระภูมิ ↑ | 77.78 ± 3.93 | 72.22 ± 10.39 |
| ECE (10 bins) ↓ | 4.94 ± 1.56 | 4.53 ± 1.67 |

ResNet50/Full มีค่าเฉลี่ยสูงสุดทั้ง 4 metrics รวม และมี SD ของ Accuracy, Macro recall และ Macro F1 ต่ำกว่าโมเดลอ้างอิง Recall ของพระภูมิก็สูงกว่า ข้อแลกเปลี่ยนคือ ECE สูงกว่าเล็กน้อย

เราตัดสินใจหลังดูผล test (post-hoc) และความต่างจากโมเดลอ้างอิงยังสรุปไม่ได้ทางสถิติ ข้อสรุปนี้จึงต้องยืนยันด้วยภาพจากแหล่งภายนอก เรายังคง MobileNetV3-Large/Partial เป็น reference ในการทดสอบทางสถิติ (28/30 สมมติฐาน) ตามที่ล็อกไว้ใน `selection.json` โดยไม่แก้ lock หรือครอบครัวการทดสอบเดิม

### 6.7 Confusion matrices

เราคำนวณสัดส่วนภายในแต่ละ true class ของแต่ละ seed ก่อน แล้วเฉลี่ยและหา sample SD ข้าม 5 seeds ช่องแนวทแยงคือ Recall ของคลาสนั้น สเกลสีเป็น 0–100 เท่ากันทุกภาพ

![Confusion matrix ของ ResNet50/Full](report/report_assets/latest/confusion_resnet50_full.svg)

*ภาพที่ 16 Confusion matrix ของ ResNet50/Full (ค่าเฉลี่ย % และ SD เป็น pp ข้าม 5 seeds) พระภูมิถูกทายถูกเฉลี่ย 77.8% ± 3.9 pp ซึ่งต่ำสุดในสี่คลาส*

<details>
<summary>Confusion matrices ของทุก architecture × mode (5 seeds ต่อคู่)</summary>

| Architecture | Frozen | Partial | Full |
| --- | --- | --- | --- |
| ResNet50 | [ดูภาพ](report/report_assets/latest/confusion_resnet50_frozen.svg) | [ดูภาพ](report/report_assets/latest/confusion_resnet50_partial.svg) | [ดูภาพ](report/report_assets/latest/confusion_resnet50_full.svg) |
| EfficientNet-B0 | [ดูภาพ](report/report_assets/latest/confusion_efficientnet_b0_frozen.svg) | [ดูภาพ](report/report_assets/latest/confusion_efficientnet_b0_partial.svg) | [ดูภาพ](report/report_assets/latest/confusion_efficientnet_b0_full.svg) |
| MobileNetV3-Large | [ดูภาพ](report/report_assets/latest/confusion_mobilenet_v3_large_frozen.svg) | [ดูภาพ](report/report_assets/latest/confusion_mobilenet_v3_large_partial.svg) | [ดูภาพ](report/report_assets/latest/confusion_mobilenet_v3_large_full.svg) |
| DenseNet121 | [ดูภาพ](report/report_assets/latest/confusion_densenet121_frozen.svg) | [ดูภาพ](report/report_assets/latest/confusion_densenet121_partial.svg) | [ดูภาพ](report/report_assets/latest/confusion_densenet121_full.svg) |
| ConvNeXt-Tiny | [ดูภาพ](report/report_assets/latest/confusion_convnext_tiny_frozen.svg) | [ดูภาพ](report/report_assets/latest/confusion_convnext_tiny_partial.svg) | [ดูภาพ](report/report_assets/latest/confusion_convnext_tiny_full.svg) |

[ค่ารายช่องทั้ง 15 คู่](report/confusion_mean_sd.csv) ระบุค่าเฉลี่ย (%) และ SD (pp) ที่ใช้วาดภาพ

</details>

### 6.8 Grad-CAM ของ ResNet50/Full

<!-- resnet-followup:gradcam:start -->

เราสร้าง Grad-CAM 8 ภาพจาก ResNet50/Full seed 42 ที่ชั้น `layer4[-1]` อธิบายคลาสที่โมเดลทำนาย เลือกตัวอย่างทายถูกและทายผิดแยกตามคลาส ถ้าคลาสไหนไม่มีภาพผิด ใช้ภาพที่ทายถูกแต่ confidence ต่ำสุดแทน ชุดนี้มีภาพทายถูก 6 ภาพและทายผิด 2 ภาพ Heatmap จัดแนวกลับสู่ภาพต้นฉบับหลังตัด padding

![Grad-CAM ของ ResNet50/Full](report/report_assets/latest/gradcam_resnet50_full_seed42_readable.png)

*ภาพที่ 17 ตัวอย่าง ResNet50/Full seed 42 จัดเป็น 2 คอลัมน์ × 4 แถว แต่ละกรณีมีภาพต้นฉบับ heatmap และ overlay*

<!-- resnet-followup:gradcam:end -->

ข้อสังเกต
- หลายภาพมี heatmap อยู่ที่ตัวเรือนหรือฐานศาล เช่น `san_phraphum_0065` และ `san_tayay_0078`
- ภาพที่ทายผิดก็มี heatmap อยู่ใกล้ตัวศาลเช่นกัน: `san_phraphum_0121` ถูกทายเป็นพระพรหม (96.80%) และ `san_tayay_0093` ถูกทายเป็นพระภูมิ (94.18%) การตอบสนองใกล้ตัวศาลจึงยังไม่รับประกันว่าทายถูก
- บางภาพเป็นโฆษณาที่มีศาลหลายองค์ ข้อความ หรือฉากหลังร่วมอยู่ เช่น `san_phraphrom_0051` ชุดนี้ยังแยกไม่ได้ว่าโมเดลใช้รูปทรง ข้อความ หรือบริบทมากน้อยแค่ไหน

Grad-CAM ใช้ประกอบการวิเคราะห์ ไม่ได้พิสูจน์สาเหตุของการทำนาย

<!-- generated:RESNET_CASE_TABLE -->

| Image ID (stem) | คลาสจริง | คลาสที่ทำนาย | Confidence (%) | ตัวอย่างที่เลือก |
| --- | --- | --- | ---: | --- |
| san_phraphrom_0051 | พระพรหม | พระพรหม ✓ | 99.98 | ถูก มั่นใจสูงสุดในคลาส |
| san_phraphrom_0017 | พระพรหม | พระพรหม ✓ | 83.96 | ถูก มั่นใจต่ำสุด; คลาสนี้ไม่มีภาพผิด |
| san_phraphum_0065 | พระภูมิ | พระภูมิ ✓ | 99.95 | ถูก มั่นใจสูงสุดในคลาส |
| san_phraphum_0121 | พระภูมิ | พระพรหม ✗ | 96.80 | ผิด มั่นใจสูงสุดในคลาส |
| san_tayay_0078 | ตายาย | ตายาย ✓ | 99.89 | ถูก มั่นใจสูงสุดในคลาส |
| san_tayay_0093 | ตายาย | พระภูมิ ✗ | 94.18 | ผิด มั่นใจสูงสุดในคลาส |
| tee_ju_eia_0050 | ตี่จู้เอี๊ยะ | ตี่จู้เอี๊ยะ ✓ | 100.00 | ถูก มั่นใจสูงสุดในคลาส |
| tee_ju_eia_0137 | ตี่จู้เอี๊ยะ | ตี่จู้เอี๊ยะ ✓ | 83.11 | ถูก มั่นใจต่ำสุด; คลาสนี้ไม่มีภาพผิด |

<!-- /generated:RESNET_CASE_TABLE -->

Confidence ปัดทศนิยมสองตำแหน่ง ค่า 100.00 จึงไม่ใช่การรับประกันความถูกต้อง ตัวอย่างที่คัดเลือกนี้ใช้ตรวจพฤติกรรมรายภาพ จึงไม่นำสัดส่วน 6/8 ไปประมาณ Accuracy รวม

**ภาพที่ถูกทายผิดบ่อยที่สุด** เรานับจากทั้ง 75 runs ภาพพระภูมิอยู่ในกลุ่มที่ผิดบ่อยที่สุดซ้ำ ๆ เช่น `san_phraphum_0043` ผิด 69/75 runs และ `san_phraphum_0121` ผิด 66/75 runs ตัวเลขเหล่านี้มาจากภาพชุดเดียวกันและโมเดลที่เกี่ยวข้องกัน จึงไม่ใช่ตัวอย่างอิสระ 75 ชุด ตารางสิบอันดับแรกอยู่ใน Appendix C และมีผลครบ 85 ภาพใน [error_frequency.csv](report/error_frequency.csv)

ไฟล์ประกอบ: [resnet_gradcam_cases.csv](report/resnet_gradcam_cases.csv)

*ดู Grad-CAM และ eyeball ของ MobileNetV3-Large/Partial ซึ่งเป็นหลักฐานประกอบของโมเดลอ้างอิงใน Appendix B*

### 6.9 Confidence และ calibration

![Reliability diagram ของ ResNet50/Full](report/report_assets/latest/selected_resnet50_full_calibration.svg)

*ภาพที่ 18 Reliability diagram ของ ResNet50/Full seed 42 จาก test 85 ภาพ วงกลมคือ bin ที่มีข้อมูล ขนาดตามจำนวนภาพ เส้นประคือ confidence เท่ากับ accuracy*

ResNet50/Full มี ECE เฉลี่ย 4.94% ± 1.56 pp จาก 5 seeds ใน seed 42 bin 90–100% มี 73 จาก 85 ภาพ จึงควรอ่าน bin ที่มีภาพน้อยด้วยความระวัง MobileNetV3-Large/Partial มี ECE เฉลี่ยต่ำกว่า (4.53% ± 1.67 pp) และ DenseNet121/Partial ต่ำสุด (4.44% ± 1.18 pp) การเลือก ResNet50/Full จึงยอมรับ calibration ที่ไม่ดีที่สุด เรายังไม่ได้ทดสอบนัยสำคัญของ ECE

### 6.10 Ablation ของ ResNet50/Full

<!-- resnet-followup:ablation:start -->

เราใช้สูตรและ split เดิมของ ResNet50/Full แล้วปิดทีละปัจจัย: augmentation หรือ class weights ฝึกใหม่จาก ImageNet อย่างละ 5 matched seeds รวม 10 runs และใช้ baseline เดิม 5 seeds เป็นตัวเทียบ ไม่ได้ tune ใหม่หลังถอดปัจจัย

<!-- generated:ABLATION_TABLE -->

| เงื่อนไข | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 93.65 ± 1.34 | 92.92 ± 2.07 | 92.77 ± 1.69 | 92.56 ± 1.81 |
| ไม่ใช้ augmentation | 92.71 ± 1.93 | 92.35 ± 1.92 | 91.41 ± 2.28 | 91.24 ± 2.39 |
| ไม่ใช้ class weights | 93.18 ± 2.10 | 92.77 ± 2.55 | 92.18 ± 2.22 | 91.89 ± 2.54 |

<!-- /generated:ABLATION_TABLE -->

![Ablation ของ ResNet50/Full](report/report_assets/latest/resnet_ablation_summary.svg)

*ภาพที่ 19 Accuracy และ Macro F1 ของ baseline กับสอง variants แสดง mean ± SD และแต่ละ seed ใช้ช่วงแกน 80–100*

ทุกเงื่อนไขใช้ Full จึงใช้สีเขียวและข้าวหลามตัดเหมือนกัน แยกเงื่อนไขด้วยตำแหน่งบนแกน x

<!-- resnet-followup:ablation:end -->

<!-- generated:ABLATION_DELTAS -->

| ปัจจัยที่ถอดออก | Δ Accuracy (pp) | Δ Macro F1 (pp) | Holm p: Accuracy | Holm p: Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Augmentation | -0.94 | -1.32 | 1.000 | 0.900 |
| Class weights | -0.47 | -0.67 | 1.000 | 1.000 |

<!-- /generated:ABLATION_DELTAS -->

Δ คือ variant − baseline จับคู่ด้วย seed ค่าเฉลี่ย (%) ± sample SD (pp) มาจากห้า seeds ต่อเงื่อนไข

Baseline มี Accuracy และ Macro F1 เฉลี่ยสูงกว่าทั้งสอง variants ในข้อมูลชุดนี้ แต่การทดสอบทั้ง 4 comparisons ยังไม่พบความแตกต่างอย่างมีนัยสำคัญหลังปรับ Holm ที่ระดับ 0.05 ผลต่างที่สังเกตบอกทิศทางและขนาดเฉลี่ยในรอบนี้ แต่ยังยืนยันประโยชน์ของแต่ละปัจจัยหรือความเท่ากันของเงื่อนไขไม่ได้

ผลรายคลาสมีข้อแลกเปลี่ยน: Recall ของพระภูมิลดจาก 77.78% เป็น 74.44% ในทั้งสอง variants แต่ Precision ของคลาสนี้ในกรณีไม่ใช้ class weights เพิ่มจาก 92.17% เป็น 95.00% เรายังเลือกสูตร baseline (มีทั้ง augmentation และ class weights) และไม่นำผล ablation ไปปรับ hyperparameters

Ablation นี้เป็นผลเชิงสำรวจ ทำหลังเลือก ResNet50/Full จาก test บน split ที่เคยดูแล้ว และ loss ของกรณีใช้/ไม่ใช้ class weights อยู่คนละสเกลวัตถุประสงค์

ไฟล์ประกอบ: [matched deltas](report/resnet_ablation_matched_deltas.csv) · [statistical tests](report/resnet_ablation_statistical_tests.csv) · [per-class summary](report/resnet_ablation_per_class_summary.csv)

---

## 7) Discussion and conclusions

**Mode:** ในผลรอบนี้ การเปิดให้ backbone ปรับน้ำหนักได้ให้ Macro F1 เฉลี่ยสูงกว่าการฝึกเฉพาะ classifier ผลต่างระหว่าง Full กับ Partial ขึ้นกับโมเดล ไม่มี mode เดียวที่ดีที่สุดทุกโมเดล ข้อสรุปนี้ใช้ได้ภายใต้งบ hyperparameters เท่ากัน (5 trials ต่อ setup)

**โมเดลที่เลือก:** ResNet50/Full ได้ค่าเฉลี่ยสูงสุดทั้ง Accuracy, Macro precision, Macro recall และ Macro F1 และมี Recall ของพระภูมิสูงกว่าโมเดลอ้างอิง ข้อแลกเปลี่ยนคือ calibration ไม่ใช่ดีที่สุด ความต่างจาก MobileNetV3-Large/Partial ยังสรุปไม่ได้ทางสถิติ และเราเลือกหลังดูผล test จึงยังไม่รู้ว่า ResNet50/Full จะ generalize ได้ดีที่สุดหรือไม่

**Ablation และ Grad-CAM:** Baseline ได้ค่าเฉลี่ยสูงกว่าทั้งสอง variants แต่ยังสรุปไม่ได้ทางสถิติ Grad-CAM แสดงว่าโมเดลทายผิดด้วย confidence สูงได้ แม้ heatmap อยู่ที่ตัวศาล ดังนั้นต้องดูผลรายคลาสและ calibration ควบคู่กับภาพอธิบาย

**จุดอ่อนที่ชัดที่สุด** คือพระภูมิ ทั้งสองโมเดลมี Recall ต่ำสุดในคลาสนี้

### ข้อจำกัด

1. **ขนาดและ split ของข้อมูล** Test มี 85 ภาพและเคยถูกดูในงานก่อนหน้า Mean ± SD และ paired tests สะท้อนความแปรปรวนระหว่าง seeds บนข้อมูลชุดเดียว การยืนยันต้องใช้ภาพจากแหล่งภายนอก
2. **ขอบเขตการค้นหา** ใช้ 5 configurations และ tuning seed เดียว และไม่มี SGD ในชุดที่สุ่มได้
3. **ที่มาภาพและสิ่งที่ปนอยู่ในภาพ** ไม่ทราบ URL 112 ภาพ และไม่มีบันทึกวิธีเก็บ วันที่ หรือสิทธิการใช้ครบทุกภาพ ข้อความ ลายน้ำ ฉากหลัง และสัดส่วนภาพอาจสัมพันธ์กับคลาส
4. **ขอบเขตการวิเคราะห์** Grad-CAM ใช้ 8 ตัวอย่างจาก seed 42 Ablation มี 5 seeds ต่อเงื่อนไขบน split เดียวและทำหลังเลือกโมเดล การไม่พบนัยสำคัญไม่ได้ยืนยันว่าปัจจัยไม่มีประโยชน์หรือว่าเงื่อนไขให้ผลเท่ากัน และยังไม่ได้วัด latency ภายใต้เงื่อนไขควบคุม

### งานต่อไป

1. เก็บภาพพระภูมิที่รูปทรงหรือบริบทสับสนเพิ่ม
2. ประเมินโมเดลด้วยภาพจากแหล่งใหม่ โดยกำหนดวิธีฝึกและเกณฑ์เลือกล่วงหน้า

---

## Appendix

### A) การทำซ้ำผล follow-up

`resnet_followup.py` รันครบแล้ว (Grad-CAM, ablation 10 runs, analysis และการนำผลเข้า README) คำสั่งด้านล่างใช้ทำซ้ำจากโฟลเดอร์โครงการ ใน environment เดิมที่มี PyTorch/torchvision ไม่ต้องฝึกใหม่เพื่ออ่านรายงานนี้

```powershell
python resnet_followup.py --stage gradcam
python resnet_followup.py --stage lock
python resnet_followup.py --stage ablation --resume --wandb-mode online --wandb-project thai-shrine-classifier-td
python resnet_followup.py --stage analyze
python resnet_followup.py --stage report
```

| Stage | หน้าที่ |
|---|---|
| `gradcam` | สร้าง Grad-CAM จาก checkpoint เดิม ไม่ฝึกใหม่ |
| `lock` | ตรวจ baseline, config, split และบันทึกแผน ablation |
| `ablation` | ฝึกเพิ่ม 10 runs (`--resume` ใช้ซ้ำเฉพาะ run ที่เสร็จและไฟล์ตรงกัน) |
| `analyze` | สรุป mean/SD ผลรายคลาส ผลต่างจับคู่ และสถิติ |
| `report` | นำรูปและตารางที่สร้างเสร็จใส่ README ในเครื่อง โดยไม่ push GitHub |

ถ้าหยุดกลางการฝึก run ที่ไม่สมบูรณ์จะเริ่มเป็น attempt ใหม่ ผลใหม่อยู่ใต้ output เดิมในโฟลเดอร์เฉพาะ และไม่แก้ final registry หรือ statistical lock เดิม

### B) Grad-CAM และ eyeball ของโมเดลอ้างอิง

<details>
<summary>หลักฐานประกอบเดิม: Grad-CAM และ eyeball ของ MobileNetV3-Large/Partial</summary>

ใช้ MobileNetV3-Large/Partial, seed 42 วิเคราะห์ 8 ภาพ โดยเลือกตัวอย่างทายถูกและทายผิดแยกตาม true class หากคลาสนั้นไม่มีภาพผิด ใช้ภาพทายถูกที่ confidence ต่ำแทน ชุดตัวอย่างประกอบด้วยภาพทายถูก 7 ภาพและทายผิด 1 ภาพ

![Grad-CAM ของโมเดลอ้างอิง](report/report_assets/latest/gradcam_reference_percent.png)

*ภาพประกอบ A1 ภาพต้นฉบับ heatmap และ overlay ของ MobileNetV3-Large/Partial seed 42 จัดวางใหม่จากภาพที่บันทึกไว้*

หลายตัวอย่างมีบริเวณตอบสนองสูงใกล้ส่วนกลางศาล หลังคา หรือฐาน กรณี `san_phraphum_0076` ถูกทายเป็นศาลตายายด้วย confidence 60.01% และมี heatmap กระจุกใกล้ศาลด้านล่างขวา บางภาพมีข้อความและลายน้ำอยู่ใกล้บริเวณตอบสนองสูงด้วย จึงยังแยกไม่ได้ว่าโมเดลใช้รูปทรงหรือข้อความมากน้อยเพียงใด Grad-CAM ช่วยตั้งข้อสังเกต แต่ไม่ได้ยืนยันสาเหตุของการทำนาย

![เปรียบเทียบป้าย ImageNet กับผลหลัง fine-tune](report/report_assets/latest/eyeball_reference_percent.png)

*ภาพประกอบ A2 ป้ายกำกับจาก ImageNet model และผลหลัง fine-tune ของ MobileNetV3-Large/Partial บนภาพเดียวกัน เช่น stupa, altar และ china cabinet เทียบกับประเภทศาลไทย*

<details>
<summary>ผลทำนายรายภาพในชุดตัวอย่าง</summary>

<!-- generated:CASE_TABLE -->

| Image ID (stem) | คลาสจริง | ผล fine-tune | Confidence (%) | ImageNet top-1 |
| --- | --- | --- | ---: | --- |
| san_phraphrom_0057 | พระพรหม | พระพรหม ✓ | 100.00 | stupa |
| san_phraphrom_0066 | พระพรหม | พระพรหม ✓ | 61.49 | stupa |
| san_phraphum_0093 | พระภูมิ | พระภูมิ ✓ | 100.00 | altar |
| san_phraphum_0076 | พระภูมิ | ตายาย ✗ | 60.01 | altar |
| san_tayay_0068 | ตายาย | ตายาย ✓ | 100.00 | boathouse |
| san_tayay_0003 | ตายาย | ตายาย ✓ | 72.19 | church |
| tee_ju_eia_0001 | ตี่จู้เอี๊ยะ | ตี่จู้เอี๊ยะ ✓ | 100.00 | china cabinet |
| tee_ju_eia_0080 | ตี่จู้เอี๊ยะ | ตี่จู้เอี๊ยะ ✓ | 86.00 | vestment |

<!-- /generated:CASE_TABLE -->

</details>

</details>

ตัวอย่างของ MobileNet และ ResNet เลือกคนละชุด ภาพประกอบเหล่านี้จึงใช้วิเคราะห์แต่ละโมเดล และยังใช้ตัดสินว่าโมเดลใดตอบสนองต่อวัตถุได้ดีกว่ากันไม่ได้

### C) ข้อมูลเพิ่มเติม

#### แหล่งภาพ

จำนวนภาพจากสิบโดเมนอันดับแรก รวมรายการที่ไม่ทราบ URL

<!-- generated:DOMAINS -->

| Source domain | จำนวนภาพ |
| --- | ---: |
| Unknown | 112 |
| down-th.img.susercontent.com | 65 |
| cy.lnwfile.com | 41 |
| brahmansiam.com | 38 |
| www.kt-spirithouse.com | 36 |
| www.korwattana.com | 35 |
| cf.shopee.co.th | 24 |
| fw.lnwfile.com | 14 |
| maneesuk.com | 14 |
| image.makewebeasy.net | 13 |

<!-- /generated:DOMAINS -->

#### จำนวน parameters

จำนวนเต็มจาก saved mode configurations

<!-- generated:PARAM_TABLE -->

| Architecture | Total parameters | Trainable Full | Trainable Frozen | Trainable Partial |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 23,516,228 | 23,516,228 | 8,196 | 14,972,932 |
| EfficientNet-B0 | 4,012,672 | 4,012,672 | 5,124 | 1,134,516 |
| MobileNetV3-Large | 4,207,156 | 4,207,156 | 1,235,204 | 3,414,668 |
| DenseNet121 | 6,957,956 | 6,957,956 | 4,100 | 2,164,228 |
| ConvNeXt-Tiny | 27,823,204 | 27,823,204 | 4,612 | 14,294,020 |

<!-- /generated:PARAM_TABLE -->

#### ผลระหว่างฝึก

<!-- generated:DYNAMICS_TABLE -->

| Setup | Mean best epoch | Mean stop epoch | Train F1 (%) | Validation F1 (%) | Loss แยกตัวช่วงท้าย (runs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50/Full | 8.4 | 15.4 | 97.07 | 94.02 | 5/5 |
| ResNet50/Frozen | 15.6 | 19.2 | 88.55 | 88.86 | 0/5 |
| ResNet50/Partial | 12.6 | 19.6 | 98.91 | 94.80 | 4/5 |
| EfficientNet-B0/Full | 17.6 | 23.6 | 99.41 | 93.76 | 1/5 |
| EfficientNet-B0/Frozen | 15.8 | 18.6 | 88.53 | 85.93 | 0/5 |
| EfficientNet-B0/Partial | 7.8 | 14.6 | 97.56 | 88.26 | 5/5 |
| MobileNetV3-Large/Full | 13.4 | 20.4 | 99.29 | 93.35 | 2/5 |
| MobileNetV3-Large/Frozen | 17.2 | 23.8 | 95.87 | 91.11 | 1/5 |
| MobileNetV3-Large/Partial | 11.4 | 17.8 | 98.82 | 94.88 | 3/5 |
| DenseNet121/Full | 12.4 | 18.8 | 99.95 | 94.42 | 2/5 |
| DenseNet121/Frozen | 21.4 | 25.0 | 90.49 | 89.63 | 0/5 |
| DenseNet121/Partial | 9.6 | 16.6 | 98.49 | 93.03 | 4/5 |
| ConvNeXt-Tiny/Full | 8.4 | 15.4 | 99.82 | 93.38 | 3/5 |
| ConvNeXt-Tiny/Frozen | 13.2 | 18.2 | 89.13 | 88.26 | 0/5 |
| ConvNeXt-Tiny/Partial | 9.6 | 15.2 | 99.17 | 93.54 | 4/5 |

<!-- /generated:DYNAMICS_TABLE -->

คอลัมน์สุดท้ายนับ runs ที่ train loss ณ epoch สุดท้ายต่ำกว่าที่ checkpoint แต่ validation loss สูงกว่า เป็นสัญญาณการแยกตัวช่วงท้ายตามนิยามนี้ ยังไม่ใช่การวินิจฉัย overfitting โดยตรง Train metrics มาจากภาพที่ทำ augmentation จึงมีเงื่อนไขต่างจาก validation

#### ภาพที่ทายผิดบ่อย

<!-- generated:ERROR_TABLE -->

| Image ID (stem) | คลาสจริง | ผิด / 75 | Reference ผิด / 5 | ResNet Full ผิด / 5 |
| --- | --- | ---: | ---: | ---: |
| san_phraphum_0043 | พระภูมิ | 69 | 2 | 5 |
| san_phraphum_0121 | พระภูมิ | 66 | 5 | 4 |
| san_phraphum_0114 | พระภูมิ | 60 | 3 | 1 |
| san_phraphum_0015 | พระภูมิ | 42 | 3 | 4 |
| san_phraphum_0076 | พระภูมิ | 42 | 3 | 2 |
| san_phraphum_0113 | พระภูมิ | 35 | 3 | 1 |
| san_tayay_0046 | ตายาย | 32 | 1 | 0 |
| san_tayay_0093 | ตายาย | 28 | 0 | 2 |
| tee_ju_eia_0061 | ตี่จู้เอี๊ยะ | 23 | 1 | 1 |
| san_phraphum_0064 | พระภูมิ | 20 | 4 | 1 |

<!-- /generated:ERROR_TABLE -->

นับบน test 85 ภาพเดิม ดูผลครบใน [error_frequency.csv](report/error_frequency.csv) จำนวนครั้งจาก 75 runs ไม่ใช่ตัวอย่างอิสระ 75 ชุด

#### บันทึกการทดลองและ CoreWeave

บันทึกการทดลองอยู่ใน [W&B: thai-shrine-classifier-td](https://wandb.ai/wichaidit-m-nida/thai-shrine-classifier-td) ส่วนเว็บไซต์ของ CoreWeave คือ [coreweave.com](https://coreweave.com/)

ไฟล์ environment บันทึกเครื่อง Windows/RTX 5060 จึงยังไม่ยืนยันว่ารันบน CoreWeave Registry ระบุ online สำหรับงานหลัก 150 runs และ ablation เพิ่ม 10 runs แต่ยังไม่ได้ตรวจ cloud artifacts ครบทุกชิ้น

#### การสร้างรูปแบบรายงานจากผลที่บันทึกไว้

คำสั่งต่อไปนี้ใช้เฉพาะ Node.js เพื่อจัดรูปและตารางจากไฟล์เดิม

```powershell
node report/generate_latest_figures.js
node report/generate_selected_model_calibration.js
node report/generate_followup_figures.js
node report/generate_context_figures.js
node report/render_readme.js
```

หลังใช้คำสั่ง `python resnet_followup.py --stage report` ซึ่ง export รูปแบบเดิม ให้ใช้ `node report/render_readme.js` เพื่อคืนรูปแบบภาษาและหน่วยของรายงานฉบับนี้ โดยอ่านเฉพาะผลที่มีอยู่

### D) ไฟล์ประกอบรายงาน

- ที่มาและการตรวจ: [report_inventory.md](report/report_inventory.md) · [report_validation.md](report/report_validation.md)
- ผลราย run และสรุป: [verified_run_metrics.csv](report/verified_run_metrics.csv) · [verified_pair_summary.csv](report/verified_pair_summary.csv) · [verified_per_class_summary.csv](report/verified_per_class_summary.csv)
- สถิติ: [verified_statistics.csv](report/verified_statistics.csv)
- ResNet50 follow-up: [resnet_followup_sources.json](report/resnet_followup_sources.json) · [resnet_followup_verification.json](report/resnet_followup_verification.json) · [verify_resnet_followup.js](report/verify_resnet_followup.js)
- โมเดลที่เลือกและ calibration: [selected_model_report_sources.json](report/selected_model_report_sources.json) · [generate_selected_model_calibration.js](report/generate_selected_model_calibration.js)
- กราฟรอบล่าสุด: [generate_latest_figures.js](report/generate_latest_figures.js) · [latest_figure_sources.json](report/latest_figure_sources.json) · [derived_brightness.csv](report/derived_brightness.csv) · [confusion_mean_sd.csv](report/confusion_mean_sd.csv)
- Preprocessing: [generate_preprocessing_examples.js](report/generate_preprocessing_examples.js) · [preprocessing_preview_sources.json](report/preprocessing_preview_sources.json)
- Learning curves: [learning_curve_sources.json](report/learning_curve_sources.json) · [learning_curve_epoch_counts.csv](report/learning_curve_epoch_counts.csv)
- การตรวจรูปแบบ: [presentation_validation.json](report/qa/presentation_validation.json) · [evidence_preservation.json](report/qa/evidence_preservation.json)
- รูปแบบรายงาน: [readme_template.md](report/readme_template.md) · [render_readme.js](report/render_readme.js) · [figure_theme.js](report/figure_theme.js)
- ภาพบริบทที่ปรับหน่วย: [generate_context_figures.js](report/generate_context_figures.js) · [context_figure_sources.json](report/context_figure_sources.json)
- รูป follow-up ที่จัดใหม่: [generate_followup_figures.js](report/generate_followup_figures.js) · [followup_figure_sources.json](report/followup_figure_sources.json)
- ตรวจตัวเลข: [build_report.ps1](report/build_report.ps1)

ผลเชิงตัวเลขทั้งหมดมาจากการทดลองรอบ `outputs_modes_v3_td` ส่วน audit, canonical split และ ImageNet baseline บางส่วนเป็นหลักฐานเดิมที่ใช้ข้อมูลชุดเดียวกัน และมีสำเนาใน output ล่าสุด รายละเอียดการคัดผลรอบอื่นออกอยู่ใน inventory

---
