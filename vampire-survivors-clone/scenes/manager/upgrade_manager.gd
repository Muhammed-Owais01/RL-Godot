extends Node

@export var experience_manager: ExperienceManager
@export var upgrade_screen_scene: PackedScene

var current_upgrades = {}
var upgrade_pool: WeightedTable = WeightedTable.new()

var upgrade_anvil = preload('res://resources/upgrades/anvil.tres')
var upgrade_axe = preload('res://resources/upgrades/axe.tres')
var upgrade_axe_damage = preload('res://resources/upgrades/axe_damage.tres')
var upgrade_sword_rate = preload('res://resources/upgrades/sword_rate.tres')
var upgrade_sword_damage = preload('res://resources/upgrades/sword_damage.tres')
var upgrade_player_speed = preload('res://resources/upgrades/player_speed.tres')

func _ready():
	upgrade_pool.add_item(upgrade_axe, 10)
	upgrade_pool.add_item(upgrade_anvil, 10000)
	upgrade_pool.add_item(upgrade_sword_rate, 10)
	upgrade_pool.add_item(upgrade_sword_damage, 10)
	upgrade_pool.add_item(upgrade_player_speed, 5)
	
	experience_manager.level_up.connect(on_level_up)


func apply_upgrade(upgrade: AbilityUpgrade) -> void:
	var has_upgrade = current_upgrades.has(upgrade.id)
	if !has_upgrade:
		current_upgrades[upgrade.id] = {
			'resource': upgrade,
			'quantity': 1
		}
	else:
		current_upgrades[upgrade.id]['quantity'] += 1
	
	if upgrade.max_quantity > 0:
		var current_quantity = current_upgrades[upgrade.id]['quantity']
		if current_quantity == upgrade.max_quantity:
			upgrade_pool.remove_item(upgrade)
	
	update_upgrade_pool(upgrade)
	GameEvents.emit_ability_upgrade_added(upgrade, current_upgrades)

func update_upgrade_pool(chosen_upgrade: AbilityUpgrade) -> void:
	if chosen_upgrade.id == upgrade_axe.id:
		upgrade_pool.add_item(upgrade_axe_damage, 10)
	
	
func pick_upgrades() -> Array[AbilityUpgrade]:
	var chosen_upgrades: Array[AbilityUpgrade] = []
	
	for i in 2:
		if upgrade_pool.items.size() == chosen_upgrades.size():
			break
		var chosen_upgrade = upgrade_pool.pick_item(chosen_upgrades)
		chosen_upgrades.append(chosen_upgrade)
	
	return chosen_upgrades


func on_upgrade_selected(upgrade: AbilityUpgrade) -> void:
	apply_upgrade(upgrade)


func on_level_up(current_level: int):
	if is_rl_run():
		var chosen_upgrades = pick_upgrades()
		if chosen_upgrades.size() > 0:
			await get_tree().create_timer(1.0).timeout
			apply_upgrade(chosen_upgrades[0])
			print("RL auto-selected upgrade: %s" % chosen_upgrades[0].name)
		return

	var upgrade_screen_instance = upgrade_screen_scene.instantiate()
	add_child(upgrade_screen_instance)
	var chosen_upgrades = pick_upgrades()
	upgrade_screen_instance.set_ability_upgrades(chosen_upgrades as Array[AbilityUpgrade])
	upgrade_screen_instance.upgrade_selected.connect(on_upgrade_selected)

func is_rl_run() -> bool:
	for arg in OS.get_cmdline_args():
		if arg == "--rl" or arg.begins_with("--port="):
			return true
	return false
