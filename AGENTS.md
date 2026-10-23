# AGENTS.md — Thai Shrine Classifier: current execution contract

Updated: 24 October 2026, Asia/Bangkok. **This is the project's single current instruction document.** It replaces the old execution, enhancement, report, protocol and status Markdown files. Do not look for or recreate those versions. The user's later explicit request authorizes one new explanatory report Markdown, `EXPERIMENT_SUMMARY_TH.md`, describing the completed experiment and existing graphs. This report is an output artifact, not another instruction contract; preserve it alongside this file. This exception supersedes restrictions below on report Markdown creation for this requested artifact only.

## 1. User intent and execution boundary

**Completed MobileNet follow-up, 24 October 2026:** under the user's explicit execution authorization, the agent completed 10 MobileNetV3-Large/Partial ablation runs / 180 actual epochs, reusing five original final baseline seeds. The separately locked active study is `outputs_modes_v3_td/ablation/mobilenet_v3_large_partial_v2/`. Removing augmentation changes Accuracy/Macro F1 by -0.2353/-0.3975 pp; removing class weights by -0.9412/-1.1796 pp. All four adjusted p-values are 1.000; do not claim proven benefit, equivalence or non-inferiority. Independent Node verification recomputed 1,275 predictions, metrics, sample SDs, matched deltas and paired tests, and checked original/copy hashes: `report/mobilenet_ablation_verification.json`. All ten runs have local epoch histories, predictions, checkpoints, learning-curve images and two confusion images; paths/hashes are indexed in `artifact_index.json` and `report/mobilenet_ablation_artifacts.json`. W&B logged all ten online runs, but cloud dashboard completeness remains unverified; images are retained locally and in the published evidence rather than claimed as W&B media. Main plus both studies totals 170 completed training runs / 3,086 epochs; the ten v1 path-failure attempts had zero epochs and zero test evaluations. README now selects MobileNet/Partial using the resource objective, 85.48% fewer trainable parameters than ResNet/Full, the unchanged Holm comparisons and qualitative CAM evidence. The canonical notebook bundles 49 files (28 Python), with all experiment flags False; the ablation executed outside the notebook. No further training is needed for this report. The original selection lock, main registry, split and historical ResNet study remain byte-identical.

**24 October 2026 MobileNet ablation and final choice:** the user explicitly authorized the agent to prepare code and execute MobileNetV3-Large/Partial ablation, retaining logs and images. This supersedes the earlier agent-execution prohibition for this study only: two locked variants (no augmentation; no class weights), five matched seeds each, 10 fresh ImageNet runs / max 300 epochs, reusing the original five completed MobileNet/Partial final seeds. Use `resnet_followup.py --architecture mobilenet_v3_large --mode partial` with a separate `mobilenet_v3_large_partial_v1` study directory and four-comparison Holm family; do not alter the ResNet study, main registry, `selection.json`, canonical split or main 28/30 families. Save console logs, epoch histories, predictions, checkpoints and figures. Replace README section 6.10 with actual MobileNet ablation results only after completion; retain ResNet results as historical evidence. The final report choice is now MobileNetV3-Large/Partial, consistent with the original validation reference, justified by the resource objective, lower trainable parameter count and qualitative CAM localization in the absence of a detected Accuracy/F1 difference after Holm. Non-significance is not equivalence, parameters are not a measured latency/memory/energy benchmark, and CAM is not causal proof. Keep the exploratory status of analyses on this previously examined test split. No other training or additional ablations are authorized.

**MobileNet Windows-path correction:** initial `mobilenet_v3_large_partial_v1` attempts all failed at directory creation with WinError 206/3 before any epoch or test evaluation. Preserve that lock, registry and console log as infrastructure-failure evidence. Compact internal stage/run names correct the path limit; the active separately locked study is `mobilenet_v3_large_partial_v2`, now the MobileNet CLI default. The two variants, recipes, five seeds and 10-run/max-300-epoch budget are unchanged. Do not rewrite the v1 lock or count its zero-epoch attempts as completed training runs.

**23 October 2026 complete matched-error comparison:** the user's latest request supersedes the blank-counterpart presentation: show each model's overlay on all seven error images so correct and incorrect responses can be compared directly. This authorizes bounded diagnostic inference only for the four missing correct counterparts (ResNet: `san_phraphum_0064`, `san_phraphum_0114`; MobileNet: `san_phraphum_0043`, `san_phraphum_0113`), using the existing seed-42 checkpoints, saved predicted targets and preprocessing. Reuse the other ten CAMs; no training, retuning or final-result rewriting. Outputs are versioned under `outputs_modes_v3_td/gradcam/matched_error_counterparts_v1/seed42/`. The user explicitly authorized publishing the revised README in the preceding turn; preserve unrelated working changes and publish only the README and its required updated presentation/evidence files.

**23 October 2026 approved README revision:** the user explicitly authorized editing README and confirmed the new three-column Grad-CAM layout: original, ResNet50/Full overlay, MobileNetV3-Large/Partial overlay. This supersedes the earlier README editing prohibition for this revision; no new publishing request was made. The main section now shows four jointly correct examples (one per class), followed by all seven distinct seed-42 test error images (five ResNet errors, four MobileNet errors, two shared), leaving either correct model's overlay blank. `report/generate_comparison_assets.js` rebuilds these figures and a separate, non-README two-model loss figure from saved files only. Learning plots show mean selected epoch in loss, accuracy and validation F1 panels. Section 3.2 explains the predeclared architecture-based Partial boundaries, source references and explicitly defined parameter-bearing layer counts; freeze depth was not tuned. Node report checks only were run for this revision, with no new Python/inference/training. Preserve the contributor's member list, TOC and unrelated README content. The source notebook now bundles 47 source files, including the new report helper; experiment flags remain False.

**23 October 2026 notebook packaging:** the user requested the current project's code in one `.ipynb` to avoid publishing many Python files. `thai_shrine_classifier_project.ipynb` is now the canonical source bundle: 46 visible source cells include all 28 Python files plus report JavaScript/PowerShell and requirements (including the later misclassified-overlay helper below). The notebook's `project_source` cell magic restores byte-identical local modules, including original newlines, without executing them; normal module imports, Windows CLI entry points and code fingerprints remain intact. Local root runners, `shrine_classifier/`, `shrine_classifier_ext/` and `tests/` are retained but ignored/untracked in Git. `report/generate_matched_mobilenet_cam.py` remains separately tracked because README links directly to it. Keep all existing dataset/split/results/checkpoints in place; no new experiments are authorized by packaging. Every experiment flag defaults to False and no tests/inference/training run during notebook setup. Static Node checks are recorded in `report/notebook_validation.json`; full notebook execution remains unverified. When changing local source later, update the corresponding notebook source cell and its source manifest deliberately; do not change fingerprints merely to bypass resume checks. Preserve README and exclude it from agent commits/pushes while another contributor is editing it. Historical artifacts from the prior cleanup are under ignored `other/cleanup_2026-10-23/archive/`; `outputs/` retains canonical audit, split and compatible baseline only.

**23 October 2026 all-misclassified overlays:** the user explicitly requested overlay versions for every misclassified test case of ResNet50/Full and MobileNetV3-Large/Partial. This request authorized bounded Python diagnostic inference for the locked representative seed 42, with no training or retuning. Completed artifacts contain all five ResNet errors and all four MobileNet errors (nine model-image cases, seven distinct test images). Source images, predictions, checkpoints and Grad-CAM targets were verified; targets are each saved predicted class. Separate versioned outputs are under `outputs_modes_v3_td/gradcam/misclassified_overlays_v1/seed42/`, with presentation copies in `report/report_assets/latest/*all_misclassified_seed42_overlay.png`. Independent Node checks cover all 170 saved prediction rows, nine aligned finite/nonempty CAM arrays, image dimensions, .42-opacity overlay pixels and output hashes: `report/misclassified_overlays_verification.json`. Maximum saved-batch versus single-image probability difference was 0.0008946955 with identical predicted classes. Original README, selection and main registry hashes were preserved. The reusable helper is embedded in the notebook and restored locally as ignored `report/generate_misclassified_overlays.py`; `RUN["misclassified_overlays"]` defaults to False. This exception permits only the requested diagnostic inference, not new training or unrelated Python/tests. No GitHub publishing was requested for these new overlays.

**20 October 2026 ResNet follow-up:** the user approved selecting ResNet50/Full in the report after considering multiple test metrics, with explicit post-hoc/exploratory disclosure. The original validation-selected MobileNet reference, `selection.json`, 75+75 registry and statistical families remain unchanged. The user then requested a Python file for ResNet Grad-CAM and ablation. Implement `resnet_followup.py` for the user to run; do not execute Python/tests/inference/training as the agent. This request authorizes two ResNet50/Full ablation variants (no augmentation; no class weights), five matched seeds each, 10 new runs/max 300 epochs, reusing the matching five-seed final baseline. This explicit ResNet target supersedes section 9's default validation-reference target for this study only. Lock the variants and a separate four-comparison Holm family before new training/testing. Save under `outputs_modes_v3_td/gradcam/resnet50_full/` and `outputs_modes_v3_td/ablation/resnet50_full_v1/`; no retuning or baseline retraining. Report actual completion only after the user runs it. README follow-up blocks may be updated from verified completed artifacts by the new CLI's local `report` stage; no automatic GitHub push.

**Completed follow-up, 20 October 2026:** the user ran through `--stage report`. Saved evidence confirms 10 completed ablation runs/145 epochs, reusing five final baseline seeds, plus ResNet Grad-CAM for eight cases (six correct, two misclassified). Work totals 160 training runs/2,906 epochs including the main study. Removing augmentation reduces observed mean Accuracy/F1 by 0.94/1.32 percentage points; removing class weights by 0.47/0.67 points. None of the four ablation comparisons is significant after Holm (minimum adjusted p = 0.9002); do not claim proven benefit or equivalence. Independent report-only Node verification recomputed 1,275 prediction rows, summaries and tests and checked copied output hashes: `report/resnet_followup_verification.json`. Main README uses ResNet confusion/CAM/calibration; prior MobileNet CAM/eyeball remains labeled supplementary evidence. No further training/inference is needed to document these results, and agent did not run Python. Cloud dashboard completeness remains unverified.

**19 October 2026 report request:** the user explicitly authorized a professor-facing Markdown report suitable as a GitHub README, following `CODEX_DADS7202_REPORT_INSTRUCTIONS.md` while overriding its PDF deliverable. `README.md` and the supporting Markdown source, evidence inventory and validation files in `report/` are authorized report artifacts in addition to `EXPERIMENT_SUMMARY_TH.md`. Use only `outputs_modes_v3_td` numerical results; preserve historical folders and clearly disclose compatible inherited audit/baseline assets. Report-only PowerShell/Node computations/figures from saved outputs are allowed; do not run Python or new training. No GitHub publishing is authorized by the request to prepare a README.

The user wants **each architecture × fine-tuning mode to receive its own tuning**. Do not choose an architecture or parameters using Full alone and then apply its winning recipe to Frozen/Partial. Compare independently tuned architecture-mode pairs under equal search budgets.

The user initially approved 180 tuning + 75 final runs, then revised tuning to **5 trials per architecture-mode pair**: 75 tuning + 75 final runs, 150 total (maximum 4,500 epochs). This latest budget is authorized; do not ask again. Optional ablation or additional large experiments are separate requests.

Follow the user's latest instruction over this document. Reading a contract alone does not authorize unrelated work. Preserve original images, labels, reviewed duplicate groups, the canonical split and historical Full-only evidence. Explain material methodology changes. Ask only about unresolved label/split conflicts, an actual instructor requirement conflict, or an additional compute budget outside the approved scope; continue unaffected work.

**Current user instruction:** the user runs Python and experiments themselves; the agent inspects and edits code/documentation only. Do not run Python, tests, smoke or measured training without a later explicit request. On 14 October 2026, the registry in `outputs_modes_v3_td/` contains 75 completed tuning trials under W&B project `thai-shrine-classifier-td`. The user previously supplied a 46-test passing output. A subsequent lock bug was fixed: extract the winning recipe from the saved trial's nested `config`, rather than its top-level metrics. The regression fixture now matches real registry structure; this latest fix has not been executed/tested by the agent. Continue with `lock`, then `final --resume`, analysis, Grad-CAM/eyeball and report in the same output directory; do not rerun tuning for this extraction fix.

On 14 October 2026, the user encountered the same nested-config mismatch in analysis plots. Both tuning scatter plots now read learning rate and weight decay from each trial's saved `config`. A plotting regression test was added but not executed by the agent. The user subsequently ran through report completion. Read-only local verification confirms 75 completed tuning trials, 75 completed final runs, 15 pair summaries with five seeds each, 58 statistical rows with computational status `ok`, and nonempty Grad-CAM, eyeball and explanatory HTML artifacts in `outputs_modes_v3_td`. Statistical status `ok` does not mean significant. The report is `report_assets/understanding_th.html`. No further training is required for these fixes; cloud dashboard completeness and instructor compliance remain unverified.

## 2. Current filesystem and evidence

Rechecked locally on 13 October 2026:

| Item | Current role/status |
|---|---|
| `dataset_cleaned/` | Present; primary human-verified labels and original images before project padding |
| `dataset_cleaned/metadata_cleaned.csv` | Present; columns `filename,category,source_url` |
| `outputs/` | Present; canonical audit/split and historical Full-only experiments |
| `pipeline.py`, `shrine_classifier/` | Existing Full-only CLI and reusable data/model/training/tracking/analysis APIs |
| `pipeline_extensions.py`, `shrine_classifier_ext/` | Existing mode policies and diagnostic APIs; old measured extension CLI stages are retired |
| `tests/`, `requirements.txt`, `.venv/` | Present; inspect current dependencies rather than trusting old environment logs |
| `pipeline_modes.py` | Implemented; default is now five trials per architecture-mode pair. The latest source/document edits have not been retested. |
| `outputs_modes_v2/` | Existing prepare/synthetic-smoke artifacts use the previous 12-trial protocol fingerprint; no measured mode tuning/final results. Keep them as historical preflight evidence, not as input for the revised protocol. |
| `dataset_padding/`, `report/`, `outputs_extensions_v1/` | Not present at this recheck; do not claim retained artifacts from them |
| Instructor guideline/example PDFs, classmate notebook | Not present; instructor compliance and reference style remain unverified |

Historical `outputs/` has completed Full-only screening, tuning and 25 final runs. It is evidence of the earlier protocol, **not** new architecture-mode selection, tuning winners or resume cache. Old experiments that changed modes using a fixed Full recipe were retired and their local measured artifacts were deleted. See the cleanup audit JSON files in `docs/`. Local cleanup does not establish deletion of cloud W&B history.

The existing 85-image test split has already been examined. Deleting artifacts does not make it unseen. The new experiment on this split is an exploratory follow-up; an independent external holdout would be a separate future confirmation task. Do not automatically alter the split or collect replacement images.

## 3. Dataset, identifiers and canonical split

Use these fixed orders everywhere: logits, weights, metrics, plots, tracking, checkpoints and CAM labels.

```python
CLASSES = ["san_phraphrom", "san_phraphum", "san_tayay", "tee_ju_eia"]
ARCHITECTURES = ["resnet50", "efficientnet_b0", "mobilenet_v3_large", "densenet121", "convnext_tiny"]
MODES = ["full", "frozen", "partial"]
SEEDS = [42, 123, 456, 789, 1024]
```

| Class | Meaning | Images | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| `san_phraphrom` | ศาลพระพรหม | 103 | 70 | 17 | 16 |
| `san_phraphum` | ศาลพระภูมิ | 124 | 89 | 17 | 18 |
| `san_tayay` | ศาลตายาย | 133 | 91 | 21 | 21 |
| `tee_ju_eia` | ศาลตี่จู้เอี๊ยะ | 200 | 140 | 30 | 30 |
| Total | | 560 | 390 | 85 | 85 |

Use `image_id = class_name + '/' + filename_stem`; store actual relative path separately. Reject ID collisions, unknown classes, ambiguous metadata joins and label conflicts. Never infer labels from folder iteration. Never count padded counterparts as independent extra samples. Default data root is `dataset_cleaned`.

Preserve and verify `outputs/splits.csv`, `outputs/split_metadata.json` and the reviewed audit artifacts. Current split seed is 42, with 550 independent groups and no group overlap; every class occurs in each partition. Training seeds never change assignments. Original audit found two exact RGB duplicate pairs in `san_phraphum` (`0002/0046`, `0073/0077`); reviewed near-duplicate grouping is already reflected in the canonical split. Preserve those decisions and revalidate rather than regenerating a split.

Required split columns: `image_id,path,label,label_index,group_id,split`. Metadata records seed, grouping algorithm, target/achieved ratios, class order, manifest/group/review fingerprints and split fingerprint. Verify both CSV byte hash and canonical ID-assignment hash when copying/loading/resuming; document their definitions separately. Do not silently rewrite or regenerate any incompatible split. Fail with the specific mismatch and prerequisite.

Re-audit image decoding and metadata before measured execution. Manifest includes IDs/paths, label, dimensions, actual decoded format/mode, byte hash, EXIF-oriented pixel hash, URL/domain and audit status. Join metadata by category + stem; retain unknown URLs as unknown. Fail on corruption/missing classes/label conflicts, and save issue lists rather than silently dropping images. Existing audit recorded 448 known and 112 unknown URLs, 206 image dimensions, RGB/RGBA/CMYK/palette modes and JPEG/PNG/WEBP/GIF actual formats; file extensions are not format authority.

Ignore and report metadata/`.DS_Store` as non-images. Group identical bytes/pixels and identical nonempty source-image URLs transitively. Domains alone do not prove duplication. New perceptual-hash matches are candidates requiring review, not automatic confirmed duplicates. Produce candidate distances/contact sheets before requesting any needed human review. Inspect train/validation examples for watermarks, text, backgrounds and aspect-ratio shortcuts. Integrity checks may cover all files; model test predictions and test-driven error analysis wait until final lock. If new leakage invalidates the split/results, report it and use an explicitly versioned correction rather than hiding it.

## 4. Shared preprocessing, augmentation and loss

Primary flow: decode actual format → EXIF transpose → consistent RGB conversion → aspect-ratio-preserving resize → centered neutral square canvas → train-only mild augmentation → tensor → ImageNet normalization.

- Input size 224; fit longest side with high-quality fixed interpolation. Neutral canvas/fill RGB `(124,116,104)`; mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`.
- Composite alpha/palette transparency on neutral color; consistently convert CMYK; use documented first-frame policy for animated inputs. Preserve originals. Save resize/content bounding boxes for CAM alignment.
- Never stretch to square or automatically remove near-black content from cleaned images. Neutral padding can reduce border contrast but does not eliminate shortcuts.
- Train augmentation: rotation 12°, translation ±5%, scale `.92–1.08`, mild perspective distortion `.10` with probability `.20`, horizontal flip `.5`, brightness/contrast/saturation jitter `.10`, hue `.02`, Random Erasing `.05` with area `.02–.08`. Use neutral fill for geometric transforms and erasing; verify tensor-space erasing corresponds to neutral RGB. No vertical/90°/180° flips or structure-removing crops. Record actual numeric settings.
- Validation/test transforms are deterministic; verify repeated tensors agree. Do not fit preprocessing or weights using validation/test labels.
- Class-weighted cross entropy: `weight[c] = n_train / (4 * n_train_class[c])` in fixed class order. Save train counts/weights. Per-item weighted loss; epoch loss is sum of per-item losses divided by image count, including unequal last batches. Do not add oversampling without an explicitly named study.

Padded input is optional and currently unavailable. Do not redownload or regenerate it automatically. If explicitly requested later, pair by ID on the same split; external artificial-border removal needs reviewed contiguous-edge detection, configurable threshold/tolerance, saved crop coordinates/retained area/uncertainty and before/after grids. Do not use arbitrary foreground boxes or treat real dark pixels as padding.

## 5. Model and mode policies

Use official torchvision ImageNet pretrained weight enums; centralize constructors, feature sizes, final output replacement and spatial CAM layers. Preserve existing pretrained head structure except its final layer becomes four logits. No silent random-weight fallback in measured runs and no unrequested MLP/dropout/transformer architecture search. Record resolved enum/library versions and frozen/trainable counts.

**Apply mode before creating the optimizer in every tuning trial and every final run**, and reapply its train/eval policy after every `model.train()` call, including calls inside common training phases.

| Mode | Trainable | State during training |
|---|---|---|
| `full` | Entire backbone and classifier | Normal train state, including trainable BatchNorm buffers |
| `frozen` | Entire classifier head | Backbone frozen + eval; head train |
| `partial` | Final semantic backbone stage + classifier head | Other backbone modules frozen + eval; permitted stage/head train |

| Architecture | Required partial stage |
|---|---|
| ResNet50 | `layer4` |
| DenseNet121 | `features.denseblock4` and `features.norm5` |
| ConvNeXt-Tiny | Final ConvNeXt feature stage; verify exact installed names |
| EfficientNet-B0 | Final MBConv stage and final projection |
| MobileNetV3-Large | Final semantic InvertedResidual stage/group and final projection; inspect all group boundaries, not simply the final residual block |

Verify installed module structure and log exact qualified names, parameter membership/counts and BatchNorm/dropout policy. Test actual gradients, parameter updates and buffers; counts alone do not establish freezing. Existing extension APIs are reusable only after verifying these invariants and semantic stage definitions.

Seed Python, NumPy, PyTorch CPU/CUDA, DataLoader generator/workers. Record deterministic settings, AMP, worker count, Python/torch/torchvision versions, device/GPU/CUDA, timestamps, code hash and Git revision if available. Support Windows multiprocessing safely and CPU/Colab without hard-coded Drive paths. Inspect `.venv` and current GPU; historical environment JSON can be stale.

## 6. Independent tuning, lock and final evaluation

There are **15 architecture-mode pairs**. No Full-only screening gate may exclude a pair. Optional common-recipe screening is diagnostic only and is outside the mandatory approved run count.

| Stage | Runs | Maximum epochs/run | Maximum epochs |
|---|---:|---:|---:|
| Tuning | 5 × 3 × 5 = 75 | 30 | 2,250 |
| Final | 15 × 5 = 75 | 30 | 2,250 |
| Mandatory total | 150 | | 4,500 |

`--trials` / `--trials-per-pair` means trials per pair; `--runs` means final seeds per pair. Current defaults are all five architectures, all three modes, 5 trials, five matched seeds and 30 maximum epochs. Report this maximum and available resources/disk before long execution. Early stopping may lower actual epochs. Do not silently change the agreed budget or disable pretrained weights.

Shared predeclared search space:

| Parameter | Space |
|---|---|
| Learning rate | Log-uniform `[1e-5,1e-3]` |
| Weight decay | Log-uniform `[1e-6,1e-2]` |
| Optimizer | `adamw,adam,sgd` |
| Batch size | `16,32` |

Use tuning/training seed 42 fixed within each search, save sampler seed/configs and actual optimizer defaults (including SGD momentum). Each pair receives its own search and independently selected recipe. Never inherit a Full winner for Frozen/Partial. If validation evidence indicates an unsuitable range/boundary, report it and lock any explicit protocol revision before test; no test-driven tuning.

Online default is 15 separate W&B Sweeps, one trial per run, or a correctly logged equivalent tuner. Offline/disabled fallback is reproducible local search; offline W&B alone is not a cloud sweep service. Save all sampled successful/failed trials, tuner/backend and bounded fallback transitions. Infrastructure retries reuse trial identity/config with a new attempt and do not count as additional unique trials. No fabricated scores or sweep URLs. Every pair needs a real successful winner; missing pairs block lock/final.

Shared training settings: max epochs 30; early-stop patience 7 with `min_delta=1e-4`; `ReduceLROnPlateau(mode='max', factor=.5, patience=3)` monitoring validation Macro F1. Keep checkpoint comparison separate from early-stop minimum improvement: select checkpoint by highest validation Macro F1, exact tie → lower validation loss → earlier epoch. `best_val_acc` is accuracy at that selected epoch; peak accuracy is a separate field.

Select a tuning winner per pair by highest selected-checkpoint validation Macro F1, ties → validation accuracy → lower validation loss → trial index. Save all 15 winners to `tuning/best_params_by_arch_mode.json`. Then freeze a `selection.json` with recipes/config hashes, preprocessing/loss/modes, seeds, budgets, planned statistical families, representative seed and the **validation-selected reference architecture-mode pair**. Choose that pair from all 15 tuned validation scores, ties → validation accuracy → lower validation loss → fixed architecture order → fixed mode order. Representative seed is 42, or first explicitly locked seed if 42 is excluded.

**Prepare, smoke, tune and lock do not create/use test loaders or read model test metrics.** Final must refuse missing/incompatible/incomplete locks. Final starts fresh from official ImageNet weights for each pair/seed, using that pair's own winning recipe. Do not initialize from a tuning checkpoint. Train all 75 runs, select/reload the validation checkpoint, then test once per run. Preserve canonical test ID order. Test ranking is descriptive and cannot change the reference, hyperparameters, checkpoint rule or analysis seed.

Checkpoint/config identity includes architecture, mode, trial/seed, attempt, selected epoch/score, class order, weight enum and dataset/split/config/protocol/code fingerprints. Filenames are unique; never overwrite another attempt. Resume only matching complete new-protocol artifacts with validated config, checkpoints, predictions and metrics. A best checkpoint alone is not exact optimizer/scheduler/RNG resume state; interrupted training restarts as a recorded new attempt unless full resume state was saved. Do not count historical `outputs/` or retired mode results toward the new budget, even if parameters match.

## 7. Metrics, statistics and diagnostic outputs

Save per-run canonical predictions (image ID, true/predicted label, four probabilities, confidence, checkpoint identity), accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, classification report, raw and row-normalized confusion matrices, and actual epoch curves. Explicitly include all four classes with consistent zero-division handling. History summary events are not extra training epochs.

Aggregate **architecture + mode** across five matched seeds using mean and sample SD (`ddof=1`). Per-class support is the fixed test count, not five times it. Never pool different modes into one architecture summary or headline only a maximum seed.

Lock these two-sided paired t-test families before test:

| Family | Comparisons | Holm family size |
|---|---|---:|
| Primary | Validation-selected pair vs 14 other pairs × Accuracy/Macro F1 | 28 |
| Secondary | All three mode-pair contrasts within each of five architectures × two metrics | 30 |

Join by seed ID, not row position. Use `scipy.stats.ttest_rel`; save pair identities/direction, paired n, mean difference, t, raw p, Holm p, alpha `.05`, adjusted significance and family ID/size. Do not reuse the historical eight-comparison family. Missing seeds, non-finite values and zero-variance differences have explicit incomplete/undefined statuses, not invented p-values. Do not shrink families after seeing results. Custom architecture/mode sets require predeclared family definitions. Explain that small-n SD/tests describe seed variation on one split, not uncertainty across newly collected datasets. Report effects and avoid unsupported significance claims.

Confidence analysis: ECE with 10 bins, correct/wrong confidence, reliability plots and representative confidence histograms, with architecture/mode/seed identity. Training dynamics: actual epoch histories, best/stop epochs, validation/train gaps and tuning plots from real logged configurations. Report missing/failing runs explicitly. These analyses cannot retune models.

Unmodified ImageNet baseline is a qualitative label-space mismatch demonstration on fixed train examples, not four-class evaluation of 1,000-class logits. Save actual top labels/confidences. Reuse compatible historical baseline/audit assets with source mapping; do not invent outputs.

Grad-CAM and eyeball/error analysis use the new validation-selected pair and locked representative seed. Choose cases deterministically: correct and misclassified per true class when available; otherwise lowest-confidence difficult cases with honest labels/reasons. Include informative confusion, occlusion, unusual views, text/watermarks and backgrounds. Save original/heatmap/overlay, IDs, true/predicted/explained target, confidence, checkpoint and selection reason. Eval with gradients enabled; use safe hooks, verify finite/nonempty heatmaps and map to original content after removing known canvas padding. Support all five architectures; frozen backbones may require input gradients for CAM. CAM is diagnostic, not causal proof. Eyeball compares the same image's unmodified ImageNet label and fine-tuned label. Test ImageNet/CAM exploration is allowed only after lock/final evaluation.

## 8. Tracking, artifacts and reporting

All new outputs, including W&B files, stay under `outputs_modes_v3/` (or an explicit new out-dir). Never overwrite historical `outputs/`. The prior `outputs_modes_v2/` has the 12-trial protocol fingerprint and must not be resumed for this revised plan; the new default output directory prevents that mismatch.

```text
outputs_modes_v3/
  protocol.json, source_manifest.json, preflight.json, environment.json
  splits.csv, split_metadata.json, run_registry.json, execution_status.json
  dataset_audit/, baseline/
  tuning/<architecture>/<mode>/<trial_attempt>/
  tuning/all_trials.csv, best_params_by_arch_mode.json
  selection.json
  final/<architecture>/<mode>/<seed_attempt>/
  final/run_metrics.csv, pair_summary.csv, per_class_summary.csv
  final/statistical_tests.csv
  confidence/, diagnostics/, gradcam/, eyeball/, report_assets/
  wandb/, smoke/
  ablation/  # only when separately requested
```

Local CSV/JSON/PNG outputs are mandatory regardless of tracking mode. Write atomically where practical; registry records completed/failed/interrupted attempts and errors. Save real stage dependencies/prerequisites. Keep execution status as JSON with checks, artifact paths, blockers, tracking state, completed counts and next command. **Keep AGENTS.md as the only project instruction Markdown; write report text/asset maps/status as HTML/JSON rather than recreating duplicate MD files.** Update this document in place for future protocol revisions.

Support `--wandb-mode online|offline|disabled`, project/entity overrides and normal credentials without printing/saving keys. Online is the measured default; disabled for bounded smoke. Bound login/network retries and retain local results on failures. Record requested/effective mode/error/fallback; offline visibility requires successful sync, and disabled has no W&B history. Do not claim cloud dashboard completeness without verifying it.

Each trainable run logs locally and to W&B when available: `epoch,train_loss,train_acc,train_macro_f1,val_loss,val_acc,val_macro_f1,learning_rate`. Use epoch as chart step. End summary includes `best_epoch,best_val_acc,best_val_macro_f1`. Only locked final/ablation adds test metrics and all class precision/recall/F1/support. Log full configuration, resolved mode modules, class weights, fingerprints, attempts, tuner/backend and actual environment. Run names include architecture/mode/trial or seed/attempt. Curves/confusion/CAM/grids/tables/statistics have matching identities and sources.

The user wants a **Thai explanation for understanding**, not an automatic repeat of the old submission PDF. Explain the task and every stage from data → sweep → pair winners → lock → final → statistics/diagnostics/CAM, what each result means and what it shows. Include readable graphs **only from actual saved compatible logs/results**, with source-file mappings, configuration/mode/seed and interpretation beside each figure. Distinguish validation selection from observed test ranking. State failures, incompleteness, exploratory test status and tracking limitations. Create a standalone explanatory HTML and measured figures/tables; no unrequested PDF/deck. Missing results are explicit, not placeholders masquerading as measurements. Exact instructor compliance requires the actual source PDF if later supplied.

## 9. Optional ablation boundary

Mandatory mode comparison is not an ablation using a shared Full winner. If separately requested, use the **new tuned validation-selected pair** and its winning recipe as baseline; remove augmentation or class weights one factor at a time, keeping architecture/mode/split/seeds/config/selection policy fixed. Reuse a matching complete five-seed final baseline. Two variants add 10 runs/max 300 epochs and a separate four-comparison Holm family. Lock variants before testing; save matched Accuracy/F1/deltas. Single-seed studies are exploratory. Padded-input studies need a separate explicit request and available pairs. Mandatory `all` excludes optional ablations.

## 10. Implementation sequence and acceptance

Build stage-based `pipeline_modes.py` with required `--stage`: `prepare,smoke,tune,lock,final,analyze,gradcam,eyeball,report,all`. No default stage silently starts training. Reuse verified existing modules, rather than duplicating an entire unrelated pipeline. `all` runs mandatory stages in order and preserves the canonical split. Old `pipeline_extensions.py` measured `prepare,ablation,finetune,report` stages remain retired; do not restore the fixed-Full-recipe experiment.

Expose `--data-root,--base-out-dir,--out-dir,--architectures,--modes,--trials-per-pair` (or clearly documented `--trials`), `--runs,--seeds,--epochs,--input-size,--seed,--split-seed,--device,--num-workers,--wandb-project,--wandb-entity,--wandb-mode,--resume,--dry-run`. Defaults match this contract. Save explicit deterministic seeds for any user-requested alternative run count; fewer than three runs is smoke/debug, not final evidence. Document runnable PowerShell/Colab commands here once verified; do not present a nonexistent CLI as already runnable.

Execution order:

1. Verify data/metadata/review/groups and canonical split fingerprints; inspect current runtime/resources and resolve installed model stage boundaries.
2. Implement common mode-aware training, independent per-pair tuning, lock/final barriers, registry/fingerprints/resume, tracking fallback, metrics/statistics/CAM/error/report stages.
3. Test deterministic split/transforms and alpha/RGB handling; all five logits/backprop; all 15 mode memberships, gradients, actual updates and BatchNorm behavior; mode before optimizer/inside every trial; no test loader in prepare/smoke/tune/lock; independent winners; checkpoint tie/min-delta rules; mismatched resume rejection; seed joins/sample SD/Holm families 28/30; tracking failure recovery and no historical-result mixing.
4. Run dry-run and bounded smoke with separate outputs, train/validation only and no real test metrics/final summaries. Random-weight synthetic checks must be explicitly labeled; measured pretrained loading must be verified without silent fallback. Correct failures before measured execution.
5. When the user requests execution, run the approved 75 trials, freeze all 15 real winners and validation reference, then run 75 fresh final models with matched seeds. Persist progress locally and recover compatible attempts.
6. Analyze completed new results, generate measured plots/statistics/confidence/dynamics/CAM/eyeball and the explanatory Thai HTML. Verify calculations, source mappings, figure readability and missing-result status. Update execution status and report actual counts/failures.

**Implementation ready** means runnable CLI/APIs, required checks/smoke and dependencies, with expensive unexecuted paths labeled. **Experiment complete** means all 15 independently tuned pairs with five trials each, 75 final runs, actual predictions/mean-SD/per-class/statistics/curves/checkpoints/diagnostics/CAM/eyeball/explanation saved and tracking visibility or limitation documented. Smoke alone is insufficient; do not claim instructor verification without its PDF.

## 11. Prompt to use after switching models

```text
อ่าน AGENTS.md ไฟล์เดียวใน C:\Users\User\Desktop\deep แล้วทำตามคำขอล่าสุดของผู้ใช้
pipeline_modes.py ต้องให้ทุก 5 architectures × full/frozen/partial tune ของตัวเอง ค่า default 5 trials ต่อคู่ = 75 tuning
งบรวมที่อนุมัติคือ 75 tuning + 75 matched final seeds = 150 runs สูงสุด 4,500 epochs
ผู้ใช้จะรัน Python และการทดลองเอง; ห้ามรัน Python/tests/smoke/training เว้นแต่ผู้ใช้สั่งภายหลังโดยตรง
ล็อก 15 winners และ reference pair ด้วย validation ก่อน test และไม่ใช้สูตร Full เดิมแทนคู่ Frozen/Partial
เก็บผลใหม่ใน outputs_modes_v3 รักษา dataset/canonical split/historical outputs
ทำ analysis, statistics, confidence, dynamics, Grad-CAM, eyeball และ HTML อธิบายภาษาไทยพร้อมกราฟจาก logs จริง
ไม่เพิ่ม optional ablation/PDF/deck อัตโนมัติ ไม่ถามอนุมัติงบเดิมซ้ำ และรายงานสถานะตามผลที่รันจริง
```

The existing `outputs_modes_v2/preflight.json` records the superseded 12-trial budget and its protocol fingerprint. The revised code defaults to `outputs_modes_v3`; start with its prepare stage before any run. Keep this document as the sole project instruction Markdown and update it in place if the protocol is revised.
