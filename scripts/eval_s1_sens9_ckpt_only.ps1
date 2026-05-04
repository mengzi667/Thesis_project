param(
  [string]$PythonExe = "D:\anaconda\envs\Thesis_project\python.exe",
  [string]$Device = "cuda",
  [int]$EvalEpisodes = 30,
  [int]$SeedStart = 41000,
  [double]$LRef = 1.0,
  [double]$ERef = 0.02653543403790981
)

$ErrorActionPreference = "Stop"
Set-Location "D:\TUD\Thesis_project"

$groups = @("g0","g1","g2","g3","g4","g5","g6","g7","g8")
$skipped = @()
$done = @()

foreach ($g in $groups) {
  $ckpt = "results/s1_sens9_${g}_train/checkpoints/ddqn_final.pt"
  if (!(Test-Path $ckpt)) {
    Write-Warning "Checkpoint not found for ${g}: $ckpt ; skip."
    $skipped += $g
    continue
  }

  $out = "results/s1_sens9_${g}_eval_main"
  Write-Host ("[EVAL] {0} -> {1}" -f $g, $out)

  & $PythonExe -m rl.evaluate `
    --policy-mode checkpoint_only `
    --checkpoint $ckpt `
    --episodes $EvalEpisodes `
    --output-dir $out `
    --device $Device `
    --seed-start $SeedStart `
    --w-l 0.5 --w-e 0.5 `
    --beta-a 0.40 --beta-c 0.12 --beta-r 0.005 `
    --l-ref $LRef --e-ref $ERef `
    --omega-arrival-dist nb2 `
    --omega-nb-phi-mode by_hour `
    --omega-nb-phi-csv data/input/nb_phi_profile.csv `
    --omega-global-scale 1.5 `
    --omega-window-start-slot 0 `
    --omega-window-end-slot 7 `
    --omega-window-scale 4.0 `
    --omega-od-target-scale 8.0

  if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $g" }
  $done += $g
}

Write-Host ""
Write-Host "[SUM] Building summary..."
& $PythonExe "scripts/summarize_s1_sens9.py"
if ($LASTEXITCODE -ne 0) { throw "Summary script failed." }

Write-Host ""
Write-Host "[DONE] Checkpoint-only sensitivity evaluation finished."
Write-Host ("Evaluated groups: {0}" -f (($done -join ", ")))
if ($skipped.Count -gt 0) {
  Write-Host ("Skipped groups (missing checkpoint): {0}" -f (($skipped -join ", ")))
}
Write-Host "Summary:"
Write-Host "  results/s1_sens9_summary/all_groups_long.csv"
Write-Host "  results/s1_sens9_summary/main_profile_rank.csv"
