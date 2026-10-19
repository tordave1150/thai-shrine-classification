$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'build_report.ps1')
$readmePath=Join-Path $root 'README.md'
$text=[IO.File]::ReadAllText($readmePath)
function Insert($name,$value){
 $tag='@@'+$name+'@@';$block="<!-- generated:$name -->`n$value`n<!-- /generated:$name -->"
 if($script:text.Contains($tag)){$script:text=$script:text.Replace($tag,$block)}else{$pattern='(?s)<!-- generated:'+ $name +' -->.*?<!-- /generated:'+ $name +' -->';$script:text=[regex]::Replace($script:text,$pattern,[Text.RegularExpressions.MatchEvaluator]{param($match) $block})}
}
function Rows { [Collections.Generic.List[object]]::new() }
$rr=[Collections.Generic.List[object]]::new()
$manifest=Get-Content (Join-Path $out 'source_manifest.json') -Raw|ConvertFrom-Json
foreach($g in ($manifest.images|Group-Object source_domain|Sort-Object Count -Descending|Select-Object -First 10)){$name=if($g.Name){$g.Name}else{'Unknown'};$rr.Add(@($name,$g.Count))}
Insert DOMAINS (Table @('Source domain','จำนวนภาพ') $rr)
$classThai=@{san_phraphrom='ศาลพระพรหม';san_phraphum='ศาลพระภูมิ';san_tayay='ศาลตายาย';tee_ju_eia='ศาลตี่จู้เอี๊ยะ'}
$rr.Clear();foreach($c in $classes){$n=@($split|Where-Object label -eq $c).Count;$rr.Add(@($c,$classThai[$c],$n,((F (100*$n/560) 'F2')+'%')))};Insert CLASS_COUNTS (Table @('Class','ประเภทศาล','จำนวนภาพ','สัดส่วน') $rr)
$rr.Clear();foreach($s in @('train','validation','test')){$group=@($split|Where-Object split -eq $s);$row=@($s,$group.Count);foreach($c in $classes){$row+=@($group|Where-Object label -eq $c).Count};$rr.Add($row)};Insert SPLIT_TABLE (Table @('Split','Total','พระพรหม','พระภูมิ','ตายาย','ตี่จู้') $rr)
$archRows=@(
 @('ResNet50','Residual blocks; fc→Linear(in_features,4)','ไม่มี head dropout เพิ่ม','Residual connections ช่วยการฝึก backbone ลึก; มีparametersมากกว่าMobileNetในโมเดลที่บันทึก'),
 @('EfficientNet-B0','MBConv; classifier[-1]→Linear(in_features,4)','คง Dropout .2','MBConv/compound design เน้นประสิทธิภาพเชิงโครงสร้าง; ไม่ได้จับเวลาbenchmarkในงานนี้'),
 @('MobileNetV3-Large','InvertedResidual; classifier[-1]→Linear(in_features,4)','คง hidden Linear/Hardswish/Dropout .2','ออกแบบสำหรับงานทรัพยากรจำกัด; มีmeanสูงหรือต่ำขึ้นกับmode ไม่ใช่ดีที่สุดทุกเงื่อนไข'),
 @('DenseNet121','Dense connections; classifier→Linear(in_features,4)','ไม่มี head dropout เพิ่ม; default dense drop_rate0','ใช้feature reuse; ข้อดีของการเชื่อมต่อไม่แทนผลเชิงสถิติ'),
 @('ConvNeXt-Tiny','ConvNeXt stages; classifier[-1]→Linear(in_features,4)','คง normalization/flatten; ไม่มี head dropoutเพิ่ม','CNN designที่ปรับให้ทันสมัย; จำนวนparametersสูงสุดใน5โมเดลนี้')
)
Insert ARCH_TABLE (Table @('Architecture','Backbone/classifier replacement','Head regularization','เหตุผลและข้อพิจารณา') $archRows)
$rr.Clear();foreach($a in $archs){$row=@($display[$a]);foreach($m in $modes){$r=$runs|Where-Object{$_.architecture-eq$a-and$_.mode-eq$m}|Select-Object -First 1;$cfg=$configs[$r.identity];if($m-eq'full'){$row+=(F $cfg.mode_stats.total_parameters 'N0')};$row+=(F $cfg.mode_stats.trainable_parameters 'N0')};$rr.Add($row)};Insert PARAM_TABLE (Table @('Architecture','Total parameters','Trainable Full','Trainable Frozen','Trainable Partial') $rr)
$search=Get-Content (Join-Path $out 'tuning/search_definition.json') -Raw|ConvertFrom-Json
$rr.Clear();$i=0;foreach($v in $search.sampled_configs){$i++;$rr.Add(@($i,(F $v.learning_rate 'G7'),(F $v.weight_decay 'G7'),$v.optimizer,$v.batch_size))};Insert SEARCH_TABLE (Table @('Trial','LR','Weight decay','Optimizer','Batch') $rr)
$winners=Get-Content (Join-Path $out 'tuning/best_params_by_arch_mode.json') -Raw|ConvertFrom-Json
$rr.Clear();foreach($p in $winners.pairs.PSObject.Properties){$v=$p.Value;$rr.Add(@($display[$v.architecture],$v.mode,$v.winning_trial,(F $v.best_val_macro_f1)))};Insert WINNER_TABLE (Table @('Architecture','Mode','Trial ที่เลือก','Validation Macro F1') $rr)
$rr.Clear();$i=0;foreach($v in ($summary|Sort-Object test_macro_f1_mean -Descending)){$i++;$rr.Add(@($i,$display[$v.architecture],$v.mode,"$(F (100*$v.test_acc_mean) 'F2') ± $(F (100*$v.test_acc_sample_sd) 'F2')","$(F $v.test_macro_precision_mean) ± $(F $v.test_macro_precision_sample_sd)","$(F $v.test_macro_recall_mean) ± $(F $v.test_macro_recall_sample_sd)","$(F $v.test_macro_f1_mean) ± $(F $v.test_macro_f1_sample_sd)"))};Insert RESULT_TABLE (Table @('Rank','Architecture','Mode','Accuracy (%)','Macro precision','Macro recall','Macro F1') $rr)
$rr.Clear();foreach($v in ($classSummary|Where-Object{($_.architecture-eq'mobilenet_v3_large'-and$_.mode-eq'partial')-or($_.architecture-eq'resnet50'-and$_.mode-eq'full')})){$rr.Add(@(($display[$v.architecture]+'/'+$v.mode),$v.class,$v.support,"$(F $v.precision_mean) ± $(F $v.precision_sample_sd)","$(F $v.recall_mean) ± $(F $v.recall_sample_sd)","$(F $v.f1_mean) ± $(F $v.f1_sample_sd)"))};Insert PERCLASS_TABLE (Table @('Pair','Class','Support','Precision','Recall','F1') $rr)
$rr.Clear();foreach($v in ($checked|Where-Object{$_.significant-or($_.reference-eq'mobilenet_v3_large::partial'-and$_.competitor-eq'resnet50::full')})){
 $a=$v.reference-split'::';$b=$v.competitor-split'::'
 $comparison=$display[$a[0]]+'/'+$a[1]+'<br>− '+$display[$b[0]]+'/'+$b[1]
 $metric=if($v.metric-eq'test_acc'){'Accuracy'}else{'Macro F1'}
 $rr.Add(@($v.family_id.Replace('primary_reference_pair_28','Primary').Replace('secondary_within_arch_mode_30','Secondary'),$comparison,$metric,(F $v.mean_difference),"[$(F $v.ci95_lower), $(F $v.ci95_upper)]",(F $v.cohen_dz 'F3'),(F $v.p_holm 'F6')))
};Insert STATS_TABLE (Table @('Family','A − B','Metric','Mean Δ','95% CI (unadjusted)','d_z','Holm p') $rr)
$rr.Clear();$dynamics=[Collections.Generic.List[object]]::new()
foreach($a in $archs){foreach($m in $modes){$bb=@();$ee=@();$tf=@();$vf=@();$late=0;foreach($r in ($runs|Where-Object{$_.architecture-eq$a-and$_.mode-eq$m})){$h=$histories[$r.identity];$best=$h|Where-Object{[int]$_.epoch-eq[int]$r.best_epoch}|Select-Object -First 1;$end=$h[-1];$bb+=[int]$r.best_epoch;$ee+=[int]$r.epochs_completed;$tf+=[double]$best.train_macro_f1;$vf+=[double]$best.val_macro_f1;if([double]$end.train_loss-lt[double]$best.train_loss-and[double]$end.val_loss-gt[double]$best.val_loss){$late++}};$rr.Add(@($display[$a],$m,(F (Mean $bb) 'F1'),(F (Mean $ee) 'F1'),(F (Mean $tf)),(F (Mean $vf)),"$late/5"));$dynamics.Add([pscustomobject]@{architecture=$a;mode=$m;mean_best_epoch=(Mean $bb);mean_stop_epoch=(Mean $ee);mean_train_f1_at_best=(Mean $tf);mean_val_f1_at_best=(Mean $vf);late_divergence_runs=$late})}}
Insert DYNAMICS_TABLE (Table @('Architecture','Mode','Mean best epoch','Mean stop epoch','Train F1@best','Val F1@best','Loss แยกตัวช่วงท้าย (runs)') $rr)
$dense=$dynamics|Where-Object{$_.architecture-eq'densenet121'-and$_.mode-eq'full'}
Insert DYNAMICS_TEXT "สำหรับ DenseNet121/Full ทั้ง5histories มี mean train F1 ณ checkpoint=$(F $dense.mean_train_f1_at_best), validation F1=$(F $dense.mean_val_f1_at_best), mean best/stop epochs=$(F $dense.mean_best_epoch 'F1')/$(F $dense.mean_stop_epoch 'F1') และ late-divergence ตามนิยามข้างต้น $($dense.late_divergence_runs)/5 runs จึงมีหลักฐานการแยกตัวช่วงท้ายในบางseed ไม่ควรสรุป underfitting จากoverallscoreเพียงอย่างเดียว ดูเส้นlossและvalidationplateauประกอบทุกseed"
$dynamics|Export-Csv (Join-Path $PSScriptRoot 'verified_training_dynamics.csv') -NoTypeInformation -Encoding utf8
$rr.Clear();foreach($v in (Import-Csv (Join-Path $out 'eyeball/eyeball_cases.csv'))){$rr.Add(@($v.image_id,$v.true_label,$v.predicted_label,(F $v.confidence),$v.imagenet_label))};Insert CASE_TABLE (Table @('Image ID','True','Fine-tuned prediction','Confidence','ImageNet top1') $rr)
$errors=@{};foreach($id in $testIDs){$errors[$id]=[pscustomobject]@{image_id=$id;true_label=($split|Where-Object image_id -eq $id).label;wrong_all75=0;wrong_reference5=0;wrong_resnetfull5=0}}
foreach($r in $runs){foreach($p in (Import-Csv (Join-Path $r.run_dir 'test_predictions.csv'))){if($p.true_label-ne$p.predicted_label){$e=$errors[$p.image_id];$e.wrong_all75++;if($r.architecture-eq'mobilenet_v3_large'-and$r.mode-eq'partial'){$e.wrong_reference5++};if($r.architecture-eq'resnet50'-and$r.mode-eq'full'){$e.wrong_resnetfull5++}}}}
$errorRows=@($errors.Values|Sort-Object wrong_all75 -Descending);$errorRows|Export-Csv (Join-Path $PSScriptRoot 'error_frequency.csv') -NoTypeInformation -Encoding utf8
$rr.Clear();foreach($e in ($errorRows|Select-Object -First 10)){$rr.Add(@($e.image_id,$e.true_label,$e.wrong_all75,$e.wrong_reference5,$e.wrong_resnetfull5))};Insert ERROR_TABLE (Table @('Image ID','True class','Wrong/75','Wrong reference/5','Wrong ResNet Full/5') $rr)
if($text-match'@@[A-Z_]+@@'){throw 'Unresolved placeholder'}
Put $readmePath $text
# Run-level map makes architecture/mode/seed/split/metrics explicit.
$runMap=foreach($stage in @('trials','final_runs')){foreach($r in $reg.$stage){[pscustomobject]@{stage=$stage;identity=$r.identity;architecture=$r.architecture;mode=$r.mode;seed=$r.seed;status=$r.status;split=if($stage-eq'trials'){'train/validation'}else{'train/validation/test'};run_dir=$r.run_dir.Substring($root.Length+1).Replace('\','/');raw_or_summary='per-run raw history, predictions(test only), structured metrics';metric_names=if($stage-eq'trials'){'train/val loss,accuracy,macro_f1'}else{'train/val loss,accuracy,macro_f1; test loss,accuracy,macro precision/recall/f1; per-class P/R/F1'};report_sections='2,3,4,5,6,7'}}}
$runMap|Export-Csv (Join-Path $PSScriptRoot 'run_evidence_map.csv') -NoTypeInformation -Encoding utf8
$sourceText="<!-- Generated mirror of ../README.md; edit the canonical README narrative. -->`n"+$text.Replace('](report/', '](')
Put (Join-Path $PSScriptRoot 'report_source.md') $sourceText
"README filled; all tables generated from latest artifacts only"
