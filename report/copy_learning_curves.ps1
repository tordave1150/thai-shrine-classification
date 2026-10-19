# Copy existing measured figures and record their sources. Does not run Python.
$ErrorActionPreference = 'Stop'
$curveWorkspace = Split-Path $PSScriptRoot -Parent
$curveOutput = Join-Path $curveWorkspace 'outputs_modes_v3_td'
$curveRegistry = Get-Content -LiteralPath (Join-Path $curveOutput 'run_registry.json') -Raw | ConvertFrom-Json
$curveArchitectures = @('resnet50','efficientnet_b0','mobilenet_v3_large','densenet121','convnext_tiny')
$curveModes = @('full','frozen','partial')
$curveSeeds = @(42,123,456,789,1024)
$curveCounts = [Collections.Generic.List[object]]::new()
$curvePairs = [Collections.Generic.List[object]]::new()
$curveFiles = [Collections.Generic.List[object]]::new()
$curveRelativeFiles = [Collections.Generic.List[string]]::new()
$curveRelativeFiles.Add('all_pairs_grid.png')

foreach ($curveArchitecture in $curveArchitectures) {
    $curveRelativeFiles.Add("$curveArchitecture/mean_sd_grid.png")
    foreach ($curveMode in $curveModes) {
        $curveRuns = @($curveRegistry.final_runs | Where-Object {
            $_.architecture -eq $curveArchitecture -and $_.mode -eq $curveMode
        })
        if ($curveRuns.Count -ne 5 -or @($curveRuns | Where-Object status -ne 'completed').Count) {
            throw "Expected five completed runs: $curveArchitecture/$curveMode"
        }
        if ((@($curveRuns.seed | Sort-Object) -join ',') -ne ($curveSeeds -join ',')) {
            throw "Unexpected seed set: $curveArchitecture/$curveMode"
        }
        # The plotting script scans every directory, rather than the registry.
        $curveRunDirectories = @(Get-ChildItem -LiteralPath (Join-Path $curveOutput "final/$curveArchitecture/$curveMode") -Directory)
        if ($curveRunDirectories.Count -ne 5 -or @(Compare-Object @($curveRuns.run_dir | Sort-Object) @($curveRunDirectories.FullName | Sort-Object)).Count) {
            throw "Plot input directories differ from registered runs: $curveArchitecture/$curveMode"
        }
        $curveEnds = @{}
        $curveHistorySources = foreach ($curveRun in $curveRuns) {
            $curveHistoryPath = Join-Path $curveRun.run_dir 'history.csv'
            $curveHistory = @(Import-Csv -LiteralPath $curveHistoryPath | Where-Object { $_.train_loss -ne '' } | Sort-Object { [int]$_.epoch } -Unique)
            $curveEpochs = @($curveHistory | ForEach-Object { [int]$_.epoch })
            $curveEnd = ($curveEpochs | Measure-Object -Maximum).Maximum
            if (($curveEpochs -join ',') -ne ((1..$curveEnd) -join ',')) { throw "Non-contiguous history: $curveHistoryPath" }
            $curveEnds[[int]$curveRun.seed] = [int]$curveEnd
            [ordered]@{
                seed = [int]$curveRun.seed
                history = $curveHistoryPath.Substring($curveWorkspace.Length+1).Replace('\','/')
                history_sha256 = (Get-FileHash -LiteralPath $curveHistoryPath -Algorithm SHA256).Hash.ToLowerInvariant()
                stop_epoch = [int]$curveEnd
                selected_epoch = [int]$curveRun.best_epoch
            }
        }
        $curveMinEnd = ($curveEnds.Values | Measure-Object -Minimum).Minimum
        $curveMaxEnd = ($curveEnds.Values | Measure-Object -Maximum).Maximum
        foreach ($curveEpoch in 1..$curveMaxEnd) {
            $curveAvailable = @($curveSeeds | Where-Object { $curveEnds[$_] -ge $curveEpoch })
            $curveCounts.Add([pscustomobject]@{
                architecture=$curveArchitecture; mode=$curveMode; epoch=$curveEpoch
                available_runs=$curveAvailable.Count; available_seeds=($curveAvailable -join ';')
                sample_sd_defined=($curveAvailable.Count -ge 2)
            })
        }
        $curvePairs.Add([ordered]@{
            architecture=$curveArchitecture; mode=$curveMode; initial_n=5
            all_five_available_through_epoch=[int]$curveMinEnd
            last_epoch=[int]$curveMaxEnd; histories=@($curveHistorySources)
        })
    }
}
$curveRelativeFiles.Add('mobilenet_v3_large/partial/mean_sd_curves.png')
foreach ($curveRelative in $curveRelativeFiles) {
    $curveSource = Join-Path $curveOutput "final/curves/$curveRelative"
    $curveTarget = Join-Path $PSScriptRoot "report_assets/learning_curves/$curveRelative"
    New-Item -ItemType Directory -Path (Split-Path $curveTarget -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $curveSource -Destination $curveTarget
    $curveSourceHash = (Get-FileHash -LiteralPath $curveSource -Algorithm SHA256).Hash.ToLowerInvariant()
    $curveTargetHash = (Get-FileHash -LiteralPath $curveTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($curveSourceHash -ne $curveTargetHash) { throw "Copy hash mismatch: $curveRelative" }
    $curveFiles.Add([ordered]@{
        source="outputs_modes_v3_td/final/curves/$curveRelative"
        report_asset="report/report_assets/learning_curves/$curveRelative"
        sha256=$curveSourceHash; copied_unchanged=$true
    })
}
$curveCounts | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'learning_curve_epoch_counts.csv') -NoTypeInformation -Encoding utf8
$curveMap = [ordered]@{
    output_root='outputs_modes_v3_td'; protocol_fingerprint=$curveRegistry.protocol_fingerprint
    plot_script='plot_learning_curves.py'
    plot_script_sha256=(Get-FileHash -LiteralPath (Join-Path $curveWorkspace 'plot_learning_curves.py') -Algorithm SHA256).Hash.ToLowerInvariant()
    script_executed_for_report=$false
    aggregation='Per-epoch available-run mean and sample SD (ddof=1); shorter histories NaN-padded; no forward fill. When n=1 the plotting script substitutes SD=0, although sample SD is undefined.'
    checkpoint_marker='Mean of the five run-specific selected epochs, using validation Macro F1.'
    files=@($curveFiles); pairs=@($curvePairs)
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'learning_curve_sources.json'), ($curveMap | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
"Copied $($curveFiles.Count) existing figures; verified 15 pairs with five registered seeds each."
