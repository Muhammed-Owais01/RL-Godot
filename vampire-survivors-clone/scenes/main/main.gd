extends Node

@export var end_screen_scene: PackedScene

@onready var player: Player = %Player

var pause_menu_scene = preload('res://scenes/ui/pause_menu.tscn')

func _ready():
	player.health_component.died.connect(on_player_died)

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed('pause'):
		add_child(pause_menu_scene.instantiate())
		get_tree().root.set_input_as_handled()

func on_player_died():
	if is_rl_run():
		return
	var end_screen_instance = end_screen_scene.instantiate()
	add_child(end_screen_instance)
	end_screen_instance.set_defeat()
	SaveManager.save()

func is_rl_run() -> bool:
	for arg in OS.get_cmdline_args():
		if arg == "--rl" or arg.begins_with("--port="):
			return true
	return false
