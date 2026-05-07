param(
    [string]$GodotExe = "C:\Users\User\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe",
    [string]$ProjectPath = "E:\Owais\UNI_Projects\RL-Godot\vampire-survivors-clone",
    [string]$Model = "checkpoints/final_model",
    [int]$Episodes = 20,
    [switch]$ShowWindow
)

Set-Location -Path $PSScriptRoot

$showFlag = ""
if ($ShowWindow) {
    $showFlag = "--show-window"
}

python -m rl_bridge.eval `
    --godot-exe "$GodotExe" `
    --project-path "$ProjectPath" `
    --model "$Model" `
    --episodes $Episodes `
    $showFlag
