$ErrorActionPreference = 'Stop'

Write-Host '=== Scenario1 NewMain Pipeline START ===' -ForegroundColor Cyan
Write-Host ('Start Time: ' + (Get-Date)) -ForegroundColor DarkCyan

Set-Location 'D:\TUD\Thesis_project'
$PyExe = 'D:\anaconda\envs\Thesis_project\python.exe'
if (-not (Test-Path $PyExe)) { throw "Python executable not found: $PyExe" }

Write-Host "`n[1/4] Training: new main profile (1.5, 4.0, 8.0)" -ForegroundColor Yellow
& $PyExe -m rl.train `
  --episodes 800 `
  --output-dir results\rl_s1_newmain_train `
  --device cuda `
  --seed-start 31000 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.40 `
  --beta-c 0.12 `
  --beta-r 0.005 `
  --l-ref 1.0 `
  --e-ref 0.02653543403790981 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour `
  --omega-nb-phi-csv data/input/nb_phi_profile.csv `
  --omega-global-scale 1.5 `
  --omega-window-start-slot 0 `
  --omega-window-end-slot 7 `
  --omega-window-scale 4.0 `
  --omega-od-target-scale 8.0
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }

Write-Host "`n[2/4] Eval: sparse profile (1.0, 2.0, 5.0)" -ForegroundColor Yellow
& $PyExe -m rl.evaluate `
  --checkpoint results\rl_s1_newmain_train\checkpoints\ddqn_final.pt `
  --episodes 400 `
  --output-dir results\rl_s1_newmain_eval_sparse `
  --device cuda `
  --seed-start 41000 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.40 `
  --beta-c 0.12 `
  --beta-r 0.005 `
  --l-ref 1.0 `
  --e-ref 0.02653543403790981 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour `
  --omega-nb-phi-csv data/input/nb_phi_profile.csv `
  --omega-global-scale 1.0 `
  --omega-window-start-slot 0 `
  --omega-window-end-slot 7 `
  --omega-window-scale 2.0 `
  --omega-od-target-scale 5.0
if ($LASTEXITCODE -ne 0) { throw 'Sparse evaluation failed.' }

Write-Host "`n[3/4] Eval: new main profile (1.5, 4.0, 8.0)" -ForegroundColor Yellow
& $PyExe -m rl.evaluate `
  --checkpoint results\rl_s1_newmain_train\checkpoints\ddqn_final.pt `
  --episodes 400 `
  --output-dir results\rl_s1_newmain_eval_main `
  --device cuda `
  --seed-start 41000 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.40 `
  --beta-c 0.12 `
  --beta-r 0.005 `
  --l-ref 1.0 `
  --e-ref 0.02653543403790981 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour `
  --omega-nb-phi-csv data/input/nb_phi_profile.csv `
  --omega-global-scale 1.5 `
  --omega-window-start-slot 0 `
  --omega-window-end-slot 7 `
  --omega-window-scale 4.0 `
  --omega-od-target-scale 8.0
if ($LASTEXITCODE -ne 0) { throw 'Main evaluation failed.' }

Write-Host "`n[4/4] Eval: new dense profile (1.8, 5.0, 10.0)" -ForegroundColor Yellow
& $PyExe -m rl.evaluate `
  --checkpoint results\rl_s1_newmain_train\checkpoints\ddqn_final.pt `
  --episodes 400 `
  --output-dir results\rl_s1_newmain_eval_dense `
  --device cuda `
  --seed-start 41000 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.40 `
  --beta-c 0.12 `
  --beta-r 0.005 `
  --l-ref 1.0 `
  --e-ref 0.02653543403790981 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour `
  --omega-nb-phi-csv data/input/nb_phi_profile.csv `
  --omega-global-scale 1.8 `
  --omega-window-start-slot 0 `
  --omega-window-end-slot 7 `
  --omega-window-scale 5.0 `
  --omega-od-target-scale 10.0
if ($LASTEXITCODE -ne 0) { throw 'Dense evaluation failed.' }

Write-Host "`n=== Scenario1 NewMain Pipeline DONE ===" -ForegroundColor Green
Write-Host ('End Time: ' + (Get-Date)) -ForegroundColor DarkGreen
