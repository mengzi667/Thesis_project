param(
  [string]$PythonExe = "D:\python\python.exe",
  [string]$Device = "cuda",
  [string]$Checkpoint = "results/rl_s2_main_train/checkpoints/ddqn_final.pt",
  [string]$CalibrationJson = "results/calib_s2_main_p95_epsgreedy.json",

  # Eval budget
  [int]$EvalEpisodes = 400,

  # Seeds
  [int]$EvalSeedSparse = 22000,
  [int]$EvalSeedMain = 21000,
  [int]$EvalSeedDense = 23000,

  # Reward defaults
  [double]$WL = 0.5,
  [double]$WE = 0.5,
  [double]$BetaA = 0.40,
  [double]$BetaC = 0.12,
  [double]$BetaR = 0.005,

  # Profiles
  [double]$SparseGlobal = 1.0,
  [double]$SparseWindow = 2.0,
  [double]$SparseOd = 5.0,

  [double]$MainGlobal = 1.5,
  [double]$MainWindow = 4.0,
  [double]$MainOd = 8.0,

  [double]$DenseGlobal = 1.8,
  [double]$DenseWindow = 5.0,
  [double]$DenseOd = 10.0,

  [int]$WindowStart = 0,
  [int]$WindowEnd = 7,
  [string]$ArrivalDist = "nb2",
  [string]$PhiMode = "by_hour",
  [string]$PhiCsv = "data/input/nb_phi_profile.csv",

  # Output dirs
  [string]$OutSparse = "results/rl_s2_eval_sparse_400",
  [string]$OutMain = "results/rl_s2_eval_main_400",
  [string]$OutDense = "results/rl_s2_eval_dense_400"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\TUD\Thesis_project"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $Checkpoint)) {
  throw "Checkpoint not found: $Checkpoint"
}
if (-not (Test-Path $CalibrationJson)) {
  throw "Calibration json not found: $CalibrationJson"
}

$Calib = Get-Content $CalibrationJson -Raw | ConvertFrom-Json
$LRef = [double]$Calib.l_ref
$ERef = [double]$Calib.e_ref

Write-Host "=== Scenario2 Eval-Only Pipeline START ===" -ForegroundColor Cyan
Write-Host ("Start Time: " + (Get-Date)) -ForegroundColor DarkCyan
Write-Host ("Checkpoint : {0}" -f $Checkpoint) -ForegroundColor DarkYellow
Write-Host ("Calib JSON : {0}" -f $CalibrationJson) -ForegroundColor DarkYellow
Write-Host ("L_ref={0}, E_ref={1}" -f $LRef, $ERef) -ForegroundColor DarkYellow
Write-Host ("Episodes   : {0}" -f $EvalEpisodes) -ForegroundColor DarkYellow

Write-Host "`n[1/3] Eval (sparse profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Checkpoint `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir $OutSparse `
  --device $Device `
  --seed-start $EvalSeedSparse `
  --w-l $WL --w-e $WE `
  --beta-a $BetaA --beta-c $BetaC --beta-r $BetaR `
  --l-ref $LRef --e-ref $ERef `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $SparseGlobal `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $SparseWindow `
  --omega-od-target-scale $SparseOd
if ($LASTEXITCODE -ne 0) { throw "Step [1/3] failed." }

Write-Host "`n[2/3] Eval (main profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Checkpoint `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir $OutMain `
  --device $Device `
  --seed-start $EvalSeedMain `
  --w-l $WL --w-e $WE `
  --beta-a $BetaA --beta-c $BetaC --beta-r $BetaR `
  --l-ref $LRef --e-ref $ERef `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $MainGlobal `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $MainWindow `
  --omega-od-target-scale $MainOd
if ($LASTEXITCODE -ne 0) { throw "Step [2/3] failed." }

Write-Host "`n[3/3] Eval (dense profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Checkpoint `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir $OutDense `
  --device $Device `
  --seed-start $EvalSeedDense `
  --w-l $WL --w-e $WE `
  --beta-a $BetaA --beta-c $BetaC --beta-r $BetaR `
  --l-ref $LRef --e-ref $ERef `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $DenseGlobal `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $DenseWindow `
  --omega-od-target-scale $DenseOd
if ($LASTEXITCODE -ne 0) { throw "Step [3/3] failed." }

Write-Host "`n=== Scenario2 Eval-Only Pipeline DONE ===" -ForegroundColor Green
Write-Host ("End Time: " + (Get-Date)) -ForegroundColor DarkGreen
