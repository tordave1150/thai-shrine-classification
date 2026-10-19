# Rebuild report evidence from saved artifacts only. No Python or training.
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$out=Join-Path $root 'outputs_modes_v3_td'
$assets=Join-Path $PSScriptRoot 'report_assets'
New-Item -ItemType Directory -Path $assets -Force | Out-Null
$culture=[Globalization.CultureInfo]::InvariantCulture
function F($x,$fmt='F4'){ ([double]$x).ToString($fmt,$culture) }
function Mean($v){ ($v | Measure-Object -Average).Average }
function SD($v){ $m=Mean $v; [math]::Sqrt((($v | ForEach-Object {([double]$_-$m)*([double]$_-$m)} | Measure-Object -Sum).Sum)/($v.Count-1)) }
function Put($path,$content){ [IO.File]::WriteAllText($path,($content -join "`n"),[Text.UTF8Encoding]::new($false)) }
function Table($headers,$rows){
 $s=[Collections.Generic.List[string]]::new();$s.Add('| '+($headers -join ' | ')+' |');$s.Add('| '+(($headers|ForEach-Object{'---'}) -join ' | ')+' |')
 foreach($row in $rows){$s.Add('| '+($row -join ' | ')+' |')};$s -join "`n"
}
$reg=Get-Content (Join-Path $out 'run_registry.json') -Raw|ConvertFrom-Json
$selection=Get-Content (Join-Path $out 'selection.json') -Raw|ConvertFrom-Json
$classes=@('san_phraphrom','san_phraphum','san_tayay','tee_ju_eia')
$archs=@('resnet50','efficientnet_b0','mobilenet_v3_large','densenet121','convnext_tiny')
$modes=@('full','frozen','partial')
$display=@{resnet50='ResNet50';efficientnet_b0='EfficientNet-B0';mobilenet_v3_large='MobileNetV3-Large';densenet121='DenseNet121';convnext_tiny='ConvNeXt-Tiny'}
$split=Import-Csv (Join-Path $out 'splits.csv')
$testIDs=@($split|Where-Object split -eq 'test'|ForEach-Object image_id)
$groupOverlap=@($split|Group-Object group_id|Where-Object{@($_.Group.split|Select-Object -Unique).Count -gt 1}).Count
$runs=@($reg.final_runs|Where-Object status -eq 'completed')
if($runs.Count-ne 75 -or @($runs.identity|Select-Object -Unique).Count-ne 75){throw 'Expected 75 unique completed final runs'}
$raw=[Collections.Generic.List[object]]::new();$perclass=[Collections.Generic.List[object]]::new()
$histories=@{};$configs=@{};$maxRawError=0.0;$maxSummaryError=0.0;$maxClassError=0.0;$maxStatError=0.0
foreach($r in $runs){
 $pred=@(Import-Csv (Join-Path $r.run_dir 'test_predictions.csv'))
 if($pred.Count-ne 85 -or (($pred.image_id -join '|')-ne($testIDs -join '|'))){throw "Test order mismatch $($r.identity)"}
 $cfg=Get-Content (Join-Path $r.run_dir 'config.json') -Raw|ConvertFrom-Json
 $configs[$r.identity]=$cfg
 $histories[$r.identity]=@(Import-Csv (Join-Path $r.run_dir 'history.csv')|Where-Object{$_.train_loss-ne ''})
 $report=Get-Content (Join-Path $r.run_dir 'run_metrics.json') -Raw|ConvertFrom-Json
 $ms=@();foreach($c in $classes){
  $tp=@($pred|Where-Object{$_.true_label-eq$c -and $_.predicted_label-eq$c}).Count
  $support=@($pred|Where-Object true_label -eq $c).Count;$positive=@($pred|Where-Object predicted_label -eq $c).Count
  $p=if($positive){$tp/$positive}else{0};$rec=$tp/$support;$f1=if($p+$rec){2*$p*$rec/($p+$rec)}else{0}
  $item=[pscustomobject]@{architecture=$r.architecture;mode=$r.mode;seed=[int]$r.seed;class=$c;support=$support;precision=$p;recall=$rec;f1=$f1}
  $perclass.Add($item);$ms+=,$item
  foreach($k in @('precision','recall','f1','support')){$maxRawError=[math]::Max($maxRawError,[math]::Abs([double]$item.$k-[double]$report.per_class.$c.$k))}
 }
 $acc=@($pred|Where-Object{$_.true_label-eq$_.predicted_label}).Count/85
 $row=[pscustomobject]@{identity=$r.identity;architecture=$r.architecture;mode=$r.mode;seed=[int]$r.seed;test_acc=$acc;test_macro_precision=(Mean $ms.precision);test_macro_recall=(Mean $ms.recall);test_macro_f1=(Mean $ms.f1)}
 foreach($k in @('test_acc','test_macro_precision','test_macro_recall','test_macro_f1')){$maxRawError=[math]::Max($maxRawError,[math]::Abs([double]$row.$k-[double]$report.$k))}
 $raw.Add($row)
}
$summary=[Collections.Generic.List[object]]::new();$classSummary=[Collections.Generic.List[object]]::new()
foreach($a in $archs){foreach($m in $modes){
 $rr=@($raw|Where-Object{$_.architecture-eq$a -and $_.mode-eq$m})
 if(($rr.seed|Sort-Object)-join ',' -ne '42,123,456,789,1024'){throw "Seed mismatch $a $m"}
 $record=[ordered]@{architecture=$a;mode=$m;completed_seeds=5}
 foreach($k in @('test_acc','test_macro_precision','test_macro_recall','test_macro_f1')){$record[$k+'_mean']=Mean $rr.$k;$record[$k+'_sample_sd']=SD $rr.$k}
 $summary.Add([pscustomobject]$record)
 foreach($c in $classes){$cc=@($perclass|Where-Object{$_.architecture-eq$a -and $_.mode-eq$m -and $_.class-eq$c});$rec=[ordered]@{architecture=$a;mode=$m;class=$c;support=$cc[0].support};foreach($k in @('precision','recall','f1')){$rec[$k+'_mean']=Mean $cc.$k;$rec[$k+'_sample_sd']=SD $cc.$k};$classSummary.Add([pscustomobject]$rec)}
}}
foreach($old in (Import-Csv (Join-Path $out 'final/pair_summary.csv'))){$new=$summary|Where-Object{$_.architecture-eq$old.architecture -and $_.mode-eq$old.mode};foreach($k in $new.PSObject.Properties.Name|Where-Object{$_ -match '_(mean|sample_sd)$'}){$maxSummaryError=[math]::Max($maxSummaryError,[math]::Abs([double]$new.$k-[double]$old.$k))}}
foreach($old in (Import-Csv (Join-Path $out 'final/per_class_summary.csv'))){$new=$classSummary|Where-Object{$_.architecture-eq$old.architecture -and $_.mode-eq$old.mode -and $_.class-eq$old.class};foreach($k in @('precision_mean','precision_sample_sd','recall_mean','recall_sample_sd','f1_mean','f1_sample_sd','support')){$maxClassError=[math]::Max($maxClassError,[math]::Abs([double]$new.$k-[double]$old.$k))}}
$stats=@(Import-Csv (Join-Path $out 'final/statistical_tests.csv'))
if(@($stats|Group-Object family_id,reference,competitor,metric|Where-Object Count -gt 1).Count){throw 'Duplicate hypotheses'}
$checked=[Collections.Generic.List[object]]::new()
foreach($s in $stats){
 $ar=$s.reference.Split('::',[StringSplitOptions]::RemoveEmptyEntries);$ac=$s.competitor.Split('::',[StringSplitOptions]::RemoveEmptyEntries)
 $r1=@($raw|Where-Object{$_.architecture-eq$ar[0] -and $_.mode-eq$ar[1]}|Sort-Object seed);$r2=@($raw|Where-Object{$_.architecture-eq$ac[0] -and $_.mode-eq$ac[1]}|Sort-Object seed)
 $d=@(0..4|ForEach-Object{[double]$r1[$_].($s.metric)-[double]$r2[$_].($s.metric)})
 $mean=Mean $d;$sd=SD $d;$t=$mean/($sd/[math]::Sqrt(5));$u=[math]::Abs($t)/[math]::Sqrt($t*$t+4)
 # Exact two-sided Student t tail at df=4: .5*(1-u)^2*(u+2).
 $p=.5*(1-$u)*(1-$u)*($u+2)
 $checked.Add([pscustomobject]@{family_id=$s.family_id;reference=$s.reference;competitor=$s.competitor;metric=$s.metric;paired_n=5;mean_difference=$mean;t=$t;p_raw=$p;p_holm=0.0;cohen_dz=($mean/$sd);ci95_lower=($mean-2.776445105*$sd/[math]::Sqrt(5));ci95_upper=($mean+2.776445105*$sd/[math]::Sqrt(5));significant=$false})
 $maxStatError=[math]::Max($maxStatError,[math]::Abs($p-[double]$s.p_value));$maxStatError=[math]::Max($maxStatError,[math]::Abs($t-[double]$s.t_statistic))
}
foreach($fam in ($checked|Group-Object family_id)){$ordered=@($fam.Group|Sort-Object p_raw);$prev=0.0;for($i=0;$i-lt$ordered.Count;$i++){$v=[math]::Min(1.0,[math]::Max($prev,($ordered.Count-$i)*$ordered[$i].p_raw));$ordered[$i].p_holm=$v;$ordered[$i].significant=$v-lt.05;$prev=$v}}
foreach($s in $stats){$r=$checked|Where-Object{$_.family_id-eq$s.family_id -and $_.reference-eq$s.reference -and $_.competitor-eq$s.competitor -and $_.metric-eq$s.metric};$maxStatError=[math]::Max($maxStatError,[math]::Abs($r.p_holm-[double]$s.p_holm))}
if([math]::Max([math]::Max($maxRawError,$maxSummaryError),[math]::Max($maxClassError,$maxStatError))-gt 1e-9){throw "Metric discrepancy exceeds 1e-9: raw=$maxRawError summary=$maxSummaryError perclass=$maxClassError stats=$maxStatError"}
$raw|Export-Csv (Join-Path $PSScriptRoot 'verified_run_metrics.csv') -NoTypeInformation -Encoding utf8
$summary|Export-Csv (Join-Path $PSScriptRoot 'verified_pair_summary.csv') -NoTypeInformation -Encoding utf8
$classSummary|Export-Csv (Join-Path $PSScriptRoot 'verified_per_class_summary.csv') -NoTypeInformation -Encoding utf8
$checked|Export-Csv (Join-Path $PSScriptRoot 'verified_statistics.csv') -NoTypeInformation -Encoding utf8
$validation=[ordered]@{output_root='outputs_modes_v3_td';final_unique_runs=$raw.Count;test_predictions=($raw.Count*85);canonical_test_order_verified=$true;group_overlap=$groupOverlap;max_prediction_vs_run_metric_error=$maxRawError;max_summary_error=$maxSummaryError;max_perclass_summary_error=$maxClassError;max_statistical_error=$maxStatError;holm_families=@(28,30);statistical_rows=$checked.Count;significant_rows=@($checked|Where-Object significant).Count;test_loss_recomputed=$false;python_executed=$false;verification_method='PowerShell; metrics from predictions; paired n=5, exact Student t df=4 tail; Holm within each predeclared family'}
Put (Join-Path $PSScriptRoot 'verification.json') ($validation|ConvertTo-Json -Depth 5)
# Full project inventory, excluding installed runtime and bytecode caches.
$allFiles=@(Get-ChildItem $root -File -Force)+@(Get-ChildItem $root -Directory -Force|Where-Object Name -NotIn @('.venv','__pycache__','.git','report')|ForEach-Object{Get-ChildItem $_.FullName -Recurse -File -Force|Where-Object FullName -NotMatch '\\__pycache__\\'})
$inventory=foreach($f in $allFiles){$relative=$f.FullName.Substring($root.Length+1).Replace('\','/');$scope=if($relative.StartsWith('outputs_modes_v3_td/')){'latest'}elseif($relative.StartsWith('dataset_cleaned/')){'dataset'}elseif($relative-match '^outputs|^wandb/'){'excluded_old_or_smoke'}else{'source_or_instruction'};$a=($archs|Where-Object{$relative.Contains($_)}|Select-Object -First 1);$seed=if($relative-match 'seed(\d+)'){$Matches[1]}else{''};[pscustomobject]@{path=$relative;bytes=$f.Length;type=$f.Extension;scope=$scope;architecture=$a;seed=$seed;evidence_kind=if($f.Extension-eq'.csv'-or$f.Extension-eq'.json'){'structured'}elseif($f.Extension-eq'.pt'){'checkpoint'}elseif($f.Extension-eq'.png'){'figure'}else{'other'}}}
$inventory|Export-Csv (Join-Path $PSScriptRoot 'evidence_inventory.csv') -NoTypeInformation -Encoding utf8
# Copy selected existing figures with GitHub-safe relative names, without modification.
$copy=@{'dataset_audit/class_distribution.png'='class_distribution.png';'dataset_audit/sample_grid.png'='sample_grid.png';'dataset_audit/preprocessing_examples.png'='preprocessing.png';'baseline/pretrained_examples.png'='imagenet_baseline.png';'tuning/tuning_validation_scatter.png'='tuning_lr.png';'tuning/weight_decay_validation_scatter.png'='tuning_weight_decay.png';'report_assets/pair_accuracy.png'='pair_accuracy.png';'report_assets/pair_macro_f1.png'='pair_macro_f1.png';'gradcam/reference_pair_cam_grid.png'='gradcam_reference.png';'eyeball/eyeball_grid.png'='eyeball_reference.png';'diagnostics/training_gaps.png'='training_gaps.png';'confidence/reliability_mobilenet_v3_large_partial_seed42.png'='reference_reliability.png'}
foreach($src in $copy.Keys){Copy-Item -LiteralPath (Join-Path $out $src) -Destination (Join-Path $assets $copy[$src])}
foreach($a in $archs){$r=$runs|Where-Object{$_.architecture-eq$a -and $_.mode-eq'full' -and $_.seed-eq42};Copy-Item (Join-Path $r.run_dir 'confusion_matrix_normalized.png') (Join-Path $assets "confusion_${a}_full_seed42.png")}
$ref=$runs|Where-Object{$_.architecture-eq'mobilenet_v3_large' -and $_.mode-eq'partial' -and $_.seed-eq42}
Copy-Item (Join-Path $ref.run_dir 'confusion_matrix_normalized.png') (Join-Path $assets 'confusion_reference_seed42.png')
# All-seed curves: real stopping lengths, no padding or artificial averaging.
foreach($metric in @('macro_f1','acc','loss')){
 $svg=[Collections.Generic.List[string]]::new();$svg.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1230" height="1420" viewBox="0 0 1230 1420"><rect width="1230" height="1420" fill="white"/>')
 $svg.Add("<text x='30' y='28' font-family='sans-serif' font-size='20'>All 75 final runs: $metric | solid=train, dashed=validation</text>")
 $colors=@('#125c9e','#d26900','#29834a','#9a4aa2','#b23c3c');$seeds=@(42,123,456,789,1024)
 for($s=0;$s-lt5;$s++){$svg.Add("<text x='$(30+$s*220)' y='56' fill='$($colors[$s])' font-family='sans-serif' font-size='14'>seed $($seeds[$s])</text>")}
 $ymax=1.0;if($metric-eq'loss'){$all=@($histories.Values|ForEach-Object{$_}|ForEach-Object{[double]$_.train_loss;[double]$_.val_loss});$ymax=[math]::Ceiling(($all|Measure-Object -Maximum).Maximum*10)/10}
 for($ai=0;$ai-lt5;$ai++){for($mi=0;$mi-lt3;$mi++){
  $x0=55+$mi*405;$y0=110+$ai*260;$w=345;$h=190
  $svg.Add("<text x='$x0' y='$($y0-12)' font-family='sans-serif' font-size='14'>$($display[$archs[$ai]]) / $($modes[$mi])</text>")
  foreach($frac in @(0,.5,1)){$yy=$y0+$h-$h*$frac;$label=F ($ymax*$frac) 'F2';$svg.Add("<path d='M $x0 $yy h $w' stroke='#ddd'/><text x='$($x0-36)' y='$($yy+4)' font-family='sans-serif' font-size='10'>$label</text>")}
  $svg.Add("<path d='M $x0 $y0 v $h h $w' fill='none' stroke='#444'/><text x='$x0' y='$($y0+$h+17)' font-size='11'>epoch 1</text><text x='$($x0+$w-45)' y='$($y0+$h+17)' font-size='11'>30</text>")
  for($s=0;$s-lt5;$s++){$r=$runs|Where-Object{$_.architecture-eq$archs[$ai]-and$_.mode-eq$modes[$mi]-and$_.seed-eq$seeds[$s]};foreach($part in @('train','val')){$key=$part+'_'+$metric;$points=foreach($v in $histories[$r.identity]){$xx=$x0+([int]$v.epoch-1)/29*$w;$yy=$y0+$h-([double]$v.$key)/$ymax*$h;"$(F $xx 'F2'),$(F $yy 'F2')"};$dash=if($part-eq'val'){"stroke-dasharray='5,3'"}else{''};$svg.Add("<polyline points='$($points -join ' ')' fill='none' stroke='$($colors[$s])' stroke-width='1.3' opacity='.8' $dash/>")}}
 }};$svg.Add('</svg>');Put (Join-Path $assets "all_seed_${metric}.svg") $svg
}
# Audit EDA distribution SVG (actual properties, not inferred brightness).
$props=@(Import-Csv (Join-Path $out 'dataset_audit/image_properties.csv'))
$svg=[Collections.Generic.List[string]]::new();$svg.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" viewBox="0 0 1100 640"><rect width="1100" height="640" fill="white"/><text x="30" y="30" font-family="sans-serif" font-size="20">Image properties | n=560 | audit of canonical dataset</text>')
$fields=@('width','height','aspect_ratio','file_bytes');$maxima=@(4500,4500,4,2000000)
for($i=0;$i-lt4;$i++){$vals=@($props|ForEach-Object{if($i-eq2){[double]$_.width/[double]$_.height}else{[double]$_.($fields[$i])}});$bins=New-Object int[] 10;foreach($v in $vals){$idx=[math]::Min(9,[math]::Floor($v/$maxima[$i]*10));$bins[$idx]++};$mx=($bins|Measure-Object -Maximum).Maximum;$x=60+($i%2)*540;$y=75+[math]::Floor($i/2)*285;$svg.Add("<text x='$x' y='$y' font-family='sans-serif' font-size='16'>$($fields[$i]) | last bin includes overflow</text>");for($b=0;$b-lt10;$b++){$height=$bins[$b]/$mx*170;$xx=$x+$b*45;$yy=$y+195-$height;$label=if($i-eq3){F ($b*$maxima[$i]/10/1000) 'F0'}else{F ($b*$maxima[$i]/10) 'G3'};$svg.Add("<rect x='$xx' y='$(F $yy)' width='38' height='$(F $height)' fill='#24496b'/><text x='$xx' y='$(F ($yy-3))' font-size='10'>$($bins[$b])</text><text x='$xx' y='$($y+215)' font-size='10'>$label</text>")};if($i-eq3){$svg.Add("<text x='$x' y='$($y+235)' font-size='12'>x-axis: file size (kB)</text>")}}
$svg.Add('</svg>');Put (Join-Path $assets 'image_properties.svg') $svg
"Verified $($raw.Count) runs, 6375 predictions; max metric error=$maxRawError; stat error=$maxStatError; inventory=$($inventory.Count) files"
