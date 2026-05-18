param(
  [string]$PythonExe = "D:\python\python.exe",
  [string]$Device = "cuda",
  [string]$CalibrationJson = "results/calib_s2_main_p95_epsgreedy.json",

  # Training/eval budget
  [int]$TrainEpisodes = 800,
  [int]$EvalEpisodes = 400,

  # Seeds
  [int]$TrainSeedStart = 31000,
  [int]$EvalSeedMain = 21000,
  [int]$EvalSeedSparse = 22000,
  [int]$EvalSeedDense = 23000,

  # Reward defaults
  [double]$WL = 0.5,
  [double]$WE = 0.5,
  [double]$BetaA = 0.40,
  [double]$BetaC = 0.12,
  [double]$BetaR = 0.005,

  # Profiles
  [double]$MainGlobal = 1.5,
  [double]$MainWindow = 4.0,
  [double]$MainOd = 8.0,

  [double]$SparseGlobal = 1.0,
  [double]$SparseWindow = 2.0,
  [double]$SparseOd = 5.0,

  [double]$DenseGlobal = 1.8,
  [double]$DenseWindow = 5.0,
  [double]$DenseOd = 10.0,

  [int]$WindowStart = 0,
  [int]$WindowEnd = 7,
  [string]$ArrivalDist = "nb2",
  [string]$PhiMode = "by_hour",
  [string]$PhiCsv = "data/input/nb_phi_profile.csv"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\TUD\Thesis_project"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $CalibrationJson)) {
  throw "Calibration json not found: $CalibrationJson"
}

$Calib = Get-Content $CalibrationJson -Raw | ConvertFrom-Json
$LRef = [double]$Calib.l_ref
$ERef = [double]$Calib.e_ref

$TrainOut = "results/rl_s2_main_train"
$Ckpt = "$TrainOut/checkpoints/ddqn_final.pt"

Write-Host "=== Scenario2 Train+Eval Pipeline START ===" -ForegroundColor Cyan
Write-Host ("Start Time: " + (Get-Date)) -ForegroundColor DarkCyan
Write-Host ("Using calibration: {0}" -f $CalibrationJson) -ForegroundColor DarkYellow
Write-Host ("L_ref={0}, E_ref={1}" -f $LRef, $ERef) -ForegroundColor DarkYellow

Write-Host "`n[1/4] Training (main profile)" -ForegroundColor Yellow
& $PythonExe -m rl.train `
  --scenario scenario2 `
  --episodes $TrainEpisodes `
  --output-dir $TrainOut `
  --device $Device `
  --seed-start $TrainSeedStart `
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
if ($LASTEXITCODE -ne 0) { throw "Step [1/4] failed." }

if (-not (Test-Path $Ckpt)) {
  throw "Checkpoint not found after training: $Ckpt"
}

Write-Host "`n[2/4] Eval (sparse profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Ckpt `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir results\rl_s2_eval_sparse `
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
if ($LASTEXITCODE -ne 0) { throw "Step [2/4] failed." }

Write-Host "`n[3/4] Eval (main profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Ckpt `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir results\rl_s2_eval_main `
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
if ($LASTEXITCODE -ne 0) { throw "Step [3/4] failed." }

Write-Host "`n[4/4] Eval (dense profile)" -ForegroundColor Yellow
& $PythonExe -m rl.evaluate `
  --scenario scenario2 `
  --checkpoint $Ckpt `
  --episodes $EvalEpisodes `
  --policy-mode three `
  --output-dir results\rl_s2_eval_dense `
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
if ($LASTEXITCODE -ne 0) { throw "Step [4/4] failed." }

Write-Host "`n=== Scenario2 Train+Eval Pipeline DONE ===" -ForegroundColor Green
Write-Host ("End Time: " + (Get-Date)) -ForegroundColor DarkGreen
