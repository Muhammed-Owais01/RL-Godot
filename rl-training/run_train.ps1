param(
    [string]$GodotExe = "C:\Users\User\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe",
    [string]$ProjectPath = "E:\Owais\UNI_Projects\RL-Godot\vampire-survivors-clone",
    [int]$NumEnvs = 1,
    [int]$MaxEpisodes = 500,
    [string]$Resume = "checkpoints/best_model",
    [float]$EntCoef = 0.03,
    [int]$NSteps = 1024,
    [float]$ClipRange = 0.15
)

Set-Location -Path $PSScriptRoot

python -m rl_bridge.train `
    --godot-exe "$GodotExe" `
    --project-path "$ProjectPath" `
    --num-envs $NumEnvs `
    --max-episodes $MaxEpisodes `
    --ent-coef $EntCoef `
    --n-steps $NSteps `
    --clip-range $ClipRange
    # --resume "$Resume" `
