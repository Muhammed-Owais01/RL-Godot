param(
    [string]$GodotExe = "C:\Users\User\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe",
    [string]$ProjectPath = "E:\Owais\UNI_Projects\RL-Godot\vampire-survivors-clone",
    [string]$Model = "checkpoints/best_model",
    [int]$Episodes = 200,
    [int]$NumEnvs = 8,
    [ValidateSet("multi", "single", "custom")]
    [string]$Profile = "multi",
    [switch]$ShowWindow
)

Set-Location -Path $PSScriptRoot

if ($Profile -eq "single") {
    $NumEnvs = 1
    $Episodes = 100
} elseif ($Profile -eq "multi") {
    $NumEnvs = 8
    $Episodes = 200
}

$showFlag = ""
if ($ShowWindow) {
    $showFlag = "--show-window"
}

python -m rl_bridge.eval `
    --godot-exe "$GodotExe" `
    --project-path "$ProjectPath" `
    --model "$Model" `
    --episodes $Episodes `
    --num-envs $NumEnvs `
    $showFlag
