param(
  [string]$PythonExe = "D:\python\python.exe",
  [string]$Device = "cuda",

  # Calibration controls
  [int]$BootstrapCalibEpisodes = 150,
  [int]$BootstrapCalibSeedStart = 31000,
  [int]$BootstrapTrainEpisodes = 200,
  [int]$EpsCalibEpisodes = 200,
  [int]$EpsCalibSeedStart = 32000,
  [double]$Quantile = 0.95,
  [double]$Epsilon = 0.20,

  # Reward defaults
  [double]$WL = 0.5,
  [double]$WE = 0.5,
  [double]$BetaA = 0.40,
  [double]$BetaC = 0.12,
  [double]$BetaR = 0.005,

  # Main profile (Scenario 2)
  [double]$GlobalScale = 1.5,
  [int]$WindowStart = 0,
  [int]$WindowEnd = 7,
  [double]$WindowScale = 4.0,
  [double]$OdTargetScale = 8.0,
  [string]$ArrivalDist = "nb2",
  [string]$PhiMode = "by_hour",
  [string]$PhiCsv = "data/input/nb_phi_profile.csv"
)

$ErrorActionPreference = "Stop"
Set-Location "D:\TUD\Thesis_project"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

$BootstrapJson = "results/calib_s2_main_p95_bootstrap.json"
$BootstrapTrainOut = "results/rl_s2_bootstrap_train"
$EpsJson = "results/calib_s2_main_p95_epsgreedy.json"
$BootstrapCkpt = "$BootstrapTrainOut/checkpoints/ddqn_final.pt"

Write-Host "=== Scenario2 Calibration Pipeline START ===" -ForegroundColor Cyan
Write-Host ("Start Time: " + (Get-Date)) -ForegroundColor DarkCyan

Write-Host "`n[1/4] Bootstrap calibration (always_offer)" -ForegroundColor Yellow
& $PythonExe -m rl.calibrate_refs `
  --scenario scenario2 `
  --episodes $BootstrapCalibEpisodes `
  --seed-start $BootstrapCalibSeedStart `
  --quantile $Quantile `
  --policy always_offer `
  --output-json $BootstrapJson `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $GlobalScale `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $WindowScale `
  --omega-od-target-scale $OdTargetScale
if ($LASTEXITCODE -ne 0) { throw "Step [1/4] failed." }

if (-not (Test-Path $BootstrapJson)) {
  throw "Bootstrap calibration output missing: $BootstrapJson"
}
$Boot = Get-Content $BootstrapJson -Raw | ConvertFrom-Json
$L0 = [double]$Boot.l_ref
$E0 = [double]$Boot.e_ref
Write-Host ("  Bootstrap refs: L_ref={0}, E_ref={1}" -f $L0, $E0) -ForegroundColor DarkYellow

Write-Host "`n[2/4] Bootstrap training for epsilon-greedy calibration checkpoint" -ForegroundColor Yellow
& $PythonExe -m rl.train `
  --scenario scenario2 `
  --episodes $BootstrapTrainEpisodes `
  --output-dir $BootstrapTrainOut `
  --device $Device `
  --seed-start $BootstrapCalibSeedStart `
  --w-l $WL --w-e $WE `
  --beta-a $BetaA --beta-c $BetaC --beta-r $BetaR `
  --l-ref $L0 --e-ref $E0 `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $GlobalScale `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $WindowScale `
  --omega-od-target-scale $OdTargetScale
if ($LASTEXITCODE -ne 0) { throw "Step [2/4] failed." }

if (-not (Test-Path $BootstrapCkpt)) {
  throw "Bootstrap checkpoint missing: $BootstrapCkpt"
}

Write-Host "`n[3/4] Epsilon-greedy calibration (final refs)" -ForegroundColor Yellow
& $PythonExe -m rl.calibrate_refs `
  --scenario scenario2 `
  --episodes $EpsCalibEpisodes `
  --seed-start $EpsCalibSeedStart `
  --quantile $Quantile `
  --policy epsilon_greedy `
  --checkpoint $BootstrapCkpt `
  --epsilon $Epsilon `
  --device $Device `
  --output-json $EpsJson `
  --omega-arrival-dist $ArrivalDist `
  --omega-nb-phi-mode $PhiMode `
  --omega-nb-phi-csv $PhiCsv `
  --omega-global-scale $GlobalScale `
  --omega-window-start-slot $WindowStart `
  --omega-window-end-slot $WindowEnd `
  --omega-window-scale $WindowScale `
  --omega-od-target-scale $OdTargetScale
if ($LASTEXITCODE -ne 0) { throw "Step [3/4] failed." }

if (-not (Test-Path $EpsJson)) {
  throw "Final calibration output missing: $EpsJson"
}
$Final = Get-Content $EpsJson -Raw | ConvertFrom-Json
$L1 = [double]$Final.l_ref
$E1 = [double]$Final.e_ref

Write-Host "`n[4/4] Calibration done" -ForegroundColor Green
Write-Host ("  Final refs: L_ref={0}, E_ref={1}" -f $L1, $E1) -ForegroundColor Green
Write-Host ("  Bootstrap JSON: {0}" -f $BootstrapJson)
Write-Host ("  Final JSON:     {0}" -f $EpsJson)
Write-Host "`n=== Scenario2 Calibration Pipeline DONE ===" -ForegroundColor Green
Write-Host ("End Time: " + (Get-Date)) -ForegroundColor DarkGreen
