param(
  [string]$PythonExe = "D:\anaconda\envs\Thesis_project\python.exe",
  [string]$Device = "cuda",
  [int]$TrainEpisodes = 150,
  [int]$EvalEpisodes = 30,
  [double]$LRef = 1.0,
  [double]$ERef = 0.02653543403790981
)

$ErrorActionPreference = "Stop"
Set-Location "D:\TUD\Thesis_project"

# Fixed common settings
$commonTrain = @(
  "--episodes", "$TrainEpisodes",
  "--device", "$Device",
  "--hidden-dim", "64",
  "--checkpoint-every", "150",
  "--transition-dump-every", "0",
  "--w-l", "0.5",
  "--w-e", "0.5",
  "--l-ref", "$LRef",
  "--e-ref", "$ERef",
  "--omega-arrival-dist", "nb2",
  "--omega-nb-phi-mode", "by_hour",
  "--omega-nb-phi-csv", "data/input/nb_phi_profile.csv",
  "--omega-global-scale", "1.5",
  "--omega-window-start-slot", "0",
  "--omega-window-end-slot", "7",
  "--omega-window-scale", "4.0",
  "--omega-od-target-scale", "8.0"
)

$commonEvalBase = @(
  "--episodes", "$EvalEpisodes",
  "--device", "$Device",
  "--w-l", "0.5",
  "--w-e", "0.5",
  "--l-ref", "$LRef",
  "--e-ref", "$ERef",
  "--omega-arrival-dist", "nb2",
  "--omega-nb-phi-mode", "by_hour",
  "--omega-nb-phi-csv", "data/input/nb_phi_profile.csv",
  "--omega-window-start-slot", "0",
  "--omega-window-end-slot", "7"
)

# 9 groups (G0-G8)
$groups = @(
  @{ id="g0"; betaA="0.40"; betaC="0.12"; betaR="0.005"; wL="0.5"; wE="0.5" },
  @{ id="g1"; betaA="0.30"; betaC="0.12"; betaR="0.005"; wL="0.5"; wE="0.5" },
  @{ id="g2"; betaA="0.50"; betaC="0.12"; betaR="0.005"; wL="0.5"; wE="0.5" },
  @{ id="g3"; betaA="0.40"; betaC="0.08"; betaR="0.005"; wL="0.5"; wE="0.5" },
  @{ id="g4"; betaA="0.40"; betaC="0.16"; betaR="0.005"; wL="0.5"; wE="0.5" },
  @{ id="g5"; betaA="0.40"; betaC="0.12"; betaR="0.005"; wL="0.7"; wE="0.3" },
  @{ id="g6"; betaA="0.40"; betaC="0.12"; betaR="0.005"; wL="0.6"; wE="0.4" },
  @{ id="g7"; betaA="0.40"; betaC="0.12"; betaR="0.005"; wL="0.4"; wE="0.6" },
  @{ id="g8"; betaA="0.40"; betaC="0.12"; betaR="0.005"; wL="0.3"; wE="0.7" }
)

# Main-only profile setting (checkpoint sensitivity in main profile)
$profiles = @(
  @{ name="main"; global="1.5"; windowScale="4.0"; odScale="8.0"; seed="41000" }
)

foreach ($g in $groups) {
  Write-Host ""
  Write-Host ("[RUN] Group {0}  (w_l={1}, w_e={2}, beta_a={3}, beta_c={4}, beta_r={5})" -f `
    $g.id, $g.wL, $g.wE, $g.betaA, $g.betaC, $g.betaR)

  $trainOut = "results/s1_sens9_{0}_train" -f $g.id
  & $PythonExe -m rl.train `
    @commonTrain `
    --output-dir $trainOut `
    --w-l $g.wL --w-e $g.wE `
    --beta-a $g.betaA --beta-c $g.betaC --beta-r $g.betaR `
    --seed-start 31000
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $($g.id)" }

  $ckpt = "$trainOut/checkpoints/ddqn_final.pt"

  foreach ($p in $profiles) {
    $evalOut = "results/s1_sens9_{0}_eval_{1}" -f $g.id, $p.name
    Write-Host ("  [EVAL] {0} -> {1}" -f $p.name, $evalOut)
    & $PythonExe -m rl.evaluate `
      @commonEvalBase `
      --policy-mode checkpoint_only `
      --checkpoint $ckpt `
      --output-dir $evalOut `
      --seed-start $p.seed `
      --result-tag main_comparable `
      --w-l $g.wL --w-e $g.wE `
      --beta-a $g.betaA --beta-c $g.betaC --beta-r $g.betaR `
      --omega-global-scale $p.global `
      --omega-window-scale $p.windowScale `
      --omega-od-target-scale $p.odScale
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($g.id) profile=$($p.name)" }
  }
}

Write-Host ""
Write-Host "[DONE] All 9 groups completed. Running summary and plots..."
& $PythonExe "scripts/summarize_s1_sens9.py"
if ($LASTEXITCODE -ne 0) { throw "Summary script failed." }
Write-Host "[DONE] Sensitivity pipeline finished."
