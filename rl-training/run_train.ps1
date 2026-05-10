param(
    [string]$GodotExe = "C:\Users\User\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe",
    [string]$ProjectPath = "E:\Owais\UNI_Projects\RL-Godot\vampire-survivors-clone",
    [ValidateSet("multi", "single", "custom")]
    [string]$Profile = "multi",
    [int]$NumEnvs = 8,
    [int]$MaxEpisodes = 500,
    [string]$Resume = "checkpoints/best_model",
    [float]$EntCoef = 0.03,
    [int]$NSteps = 128,
    [float]$ClipRange = 0.15,
    [int]$BatchSize = 128,
    [string]$Device = "cuda"
)

Set-Location -Path $PSScriptRoot

if ($Profile -eq "single") {
    $NumEnvs = 1
    $NSteps = 1024
    $BatchSize = 128
} elseif ($Profile -eq "multi") {
    $NumEnvs = 8
    $NSteps = 128
    $BatchSize = 128
}

python -m rl_bridge.train `
    --godot-exe "$GodotExe" `
    --project-path "$ProjectPath" `
    --num-envs $NumEnvs `
    --max-episodes $MaxEpisodes `
    --ent-coef $EntCoef `
    --n-steps $NSteps `
    --clip-range $ClipRange `
    --batch-size $BatchSize `
    --device $Device
    # --resume "$Resume" `
