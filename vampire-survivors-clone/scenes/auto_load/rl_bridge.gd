extends Node

const DEFAULT_PORT: int = 11008
const CONNECT_RETRY_MS: int = 500
const INITIAL_CONNECT_DELAY_MS: int = 750
const OBS_ENEMY_COUNT: int = 4
const OBS_SIZE: int = 43
const ARENA_SIZE: Vector2 = Vector2(640, 360)
const ARENA_DIAGONAL: float = 730.0

const REWARD_SURVIVAL: float = 0.02
const REWARD_DAMAGE_PER_HP: float = -0.2
const REWARD_KILL: float = 0.1
const REWARD_WAVE: float = 1.0
const REWARD_WIN: float = 50.0
const REWARD_DEATH: float = -20.0
const REWARD_INACTIVITY: float = -0.002

var client: StreamPeerTCP
var is_connected: bool = false
var connect_port: int = DEFAULT_PORT
var last_connect_attempt_ms: int = 0
var connect_ready_ms: int = 0

var arena_time_manager: ArenaTimeManager
var enemy_manager: Node
var experience_manager: Node
var upgrade_manager: Node
var player: Node2D
var dash_component: Node
var sword_controller: Node

var episode_total_reward: float = 0.0
var episode_step_count: int = 0
var episode_done: bool = false
var episode_number: int = 0

var previous_health: float = 0.0
var previous_enemy_count: int = 0
var previous_arena_difficulty: int = 0

var last_step_reward: float = 0.0
var last_action_id: int = 0

var recv_buffer: PackedByteArray = PackedByteArray()

# Stats
var action_histogram: Array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
var total_actions_received: int = 0
var best_episode_reward: float = -999.0
var best_episode_steps: int = 0

# HUD
var hud_label: Label = null
const ACTION_NAMES: Array = ["idle", "up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left", "idle2", "dash"]

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	connect_port = get_port_from_args(DEFAULT_PORT)
	connect_ready_ms = Time.get_ticks_msec() + INITIAL_CONNECT_DELAY_MS
	if is_rl_run():
		_create_hud()

func _process(_delta: float) -> void:
	if not is_connected:
		try_connect()
		return
	client.poll()
	if client.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return
	read_incoming_messages()
	_update_hud()

func connect_to_server(port: int) -> void:
	client = StreamPeerTCP.new()
	var err = client.connect_to_host("127.0.0.1", port)
	if err == OK:
		is_connected = true
		print("[RL] Connected to Python on port %d" % port)
		return

func try_connect() -> void:
	var now_ms = Time.get_ticks_msec()
	if now_ms < connect_ready_ms:
		return
	if now_ms - last_connect_attempt_ms < CONNECT_RETRY_MS:
		return
	last_connect_attempt_ms = now_ms
	connect_to_server(connect_port)

func handle_message(message: Dictionary) -> void:
	var msg_type = message.get("type", "")
	match msg_type:
		"handshake":
			return
		"env_info":
			send_env_info()
		"reset":
			await reset_episode()
			send_reset()
		"action":
			apply_action(message)
			await step_environment()
			send_step()
		"call":
			send_json({"type": "call", "returns": null})
		"close":
			get_tree().quit()
		_:
			send_json({"type": "error", "message": "Unknown message"})

func send_env_info() -> void:
	var response = {
		"type": "env_info",
		"n_agents": 1,
		"action_space": {
			"action": {
				"action_type": "discrete",
				"size": 11
			}
		},
		"observation_space": {
			"obs": {
				"space": "box",
				"size": [OBS_SIZE]
			}
		},
		"agent_policy_names": ["shared_policy"]
	}
	send_json(response)

func reset_episode() -> void:
	await _internal_reset()

func send_reset() -> void:
	var response = {
		"type": "reset",
		"obs": [get_observation()]
	}
	send_json(response)

func apply_action(message: Dictionary) -> void:
	total_actions_received += 1
	var action_list = message.get("action", [])
	if action_list.size() == 0:
		return

	var action_dict = action_list[0]
	var raw_action = action_dict.get("action", 0)
	if raw_action is Array or typeof(raw_action) == TYPE_ARRAY:
		last_action_id = int(raw_action[0]) if raw_action.size() > 0 else 0
	else:
		last_action_id = int(raw_action)

	if last_action_id >= 0 and last_action_id < action_histogram.size():
		action_histogram[last_action_id] += 1

	clear_actions()
	var move_vec = Vector2.ZERO
	var dash = false

	match last_action_id:
		0:
			pass
		1:
			move_vec = Vector2.UP
		2:
			move_vec = Vector2(1, -1).normalized()
		3:
			move_vec = Vector2.RIGHT
		4:
			move_vec = Vector2(1, 1).normalized()
		5:
			move_vec = Vector2.DOWN
		6:
			move_vec = Vector2(-1, 1).normalized()
		7:
			move_vec = Vector2.LEFT
		8:
			move_vec = Vector2(-1, -1).normalized()
		9:
			pass
		10:
			dash = true
			simulate_action("dash", true)
		_:
			pass

	if player and player.has_method("set_rl_action"):
		player.set_rl_action(move_vec, dash)

func step_environment() -> void:
	episode_step_count += 1
	last_step_reward = compute_reward()
	episode_total_reward += last_step_reward

	if is_episode_done():
		episode_done = true
		if episode_total_reward > best_episode_reward:
			best_episode_reward = episode_total_reward
			best_episode_steps = episode_step_count
		var top_action = _get_top_action()
		if episode_step_count > 1 or episode_total_reward > -1.0:
			print("[RL] Episode %d DONE | steps=%d | reward=%.2f | best=%.2f | top_action=%s" % [
				episode_number, episode_step_count, episode_total_reward, best_episode_reward, top_action
			])
		
		await _internal_reset()
		episode_done = true

func _internal_reset() -> void:
	episode_number += 1
	get_tree().reload_current_scene()
	
	var is_ready = false
	var wait_frames = 0
	while not is_ready:
		await get_tree().process_frame
		wait_frames += 1
		player = get_tree().get_first_node_in_group("player") as Node2D
		if player and is_instance_valid(player) and player.has_node("HealthComponent"):
			if player.get_node("HealthComponent").current_health > 0:
				is_ready = true
		if wait_frames > 100:
			break
			
	arena_time_manager = find_node_by_class("ArenaTimeManager")
	enemy_manager = find_node_by_name("EnemyManager")
	experience_manager = find_node_by_name("ExperienceManager")
	upgrade_manager = find_node_by_name("UpgradeManager")
	if player and player.has_node("DashComponent"):
		dash_component = player.get_node("DashComponent")
	else:
		dash_component = null
	if player and player.has_node("Abilities/SwordAbilityController"):
		sword_controller = player.get_node("Abilities/SwordAbilityController")
	else:
		sword_controller = null
	
	previous_health = get_player_health()
	previous_enemy_count = get_enemy_count()
	previous_arena_difficulty = get_arena_difficulty()
	
	episode_total_reward = 0.0
	episode_step_count = 0
	last_action_id = 0
	action_histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

func send_step() -> void:
	var response = {
		"type": "step",
		"obs": [get_observation()],
		"reward": [last_step_reward],
		"done": [episode_done]
	}
	send_json(response)
	if episode_done:
		episode_done = false

func compute_reward() -> float:
	var reward: float = 0.0
	reward += REWARD_SURVIVAL

	var current_health = get_player_health()
	var health_delta = previous_health - current_health
	if health_delta > 0:
		reward += REWARD_DAMAGE_PER_HP * health_delta
	previous_health = current_health

	var current_enemy_count = get_enemy_count()
	var killed = max(previous_enemy_count - current_enemy_count, 0)
	reward += REWARD_KILL * float(killed)
	previous_enemy_count = current_enemy_count

	var current_difficulty = get_arena_difficulty()
	var diff_increase = max(current_difficulty - previous_arena_difficulty, 0)
	if diff_increase > 0:
		reward += REWARD_WAVE * float(diff_increase)
	previous_arena_difficulty = current_difficulty

	if not is_moving() and current_enemy_count < 2:
		reward += REWARD_INACTIVITY

	if is_episode_done():
		if is_player_dead():
			reward += REWARD_DEATH
		elif arena_time_manager and arena_time_manager.has_method("is_finished"):
			if arena_time_manager.call("is_finished"):
				reward += REWARD_WIN

	return reward

func get_observation() -> Dictionary:
	var obs: Array = []

	if not player or not is_instance_valid(player):
		for i in range(OBS_SIZE):
			obs.append(0.0)
		return {"obs": obs}

	var player_pos = player.global_position / ARENA_SIZE
	obs.append(clamp(player_pos.x, -1.0, 1.0))
	obs.append(clamp(player_pos.y, -1.0, 1.0))

	var velocity = Vector2.ZERO
	if player.has_node("VelocityComponent"):
		velocity = player.get_node("VelocityComponent").velocity / 100.0
	obs.append(clamp(velocity.x, -1.0, 1.0))
	obs.append(clamp(velocity.y, -1.0, 1.0))

	obs.append(clamp(get_player_health_percent(), 0.0, 1.0))

	obs.append(float(get_arena_difficulty()) / 20.0)
	if arena_time_manager and arena_time_manager.has_method("get_time_elapsed"):
		obs.append(clamp(float(arena_time_manager.call("get_time_elapsed")) / 600.0, 0.0, 1.0))
	else:
		obs.append(0.0)
	var waves_cleared = float(min(get_arena_difficulty(), 12)) / 12.0
	obs.append(clamp(waves_cleared, 0.0, 1.0))

	var can_dash_val = 0.0
	var dashing_val = 0.0
	var dash_cooldown_ratio = 0.0
	if dash_component:
		can_dash_val = 1.0 if dash_component.can_dash else 0.0
		dashing_val = 1.0 if dash_component.dashing else 0.0
		if dash_component.has_method("get_cooldown_ratio"):
			dash_cooldown_ratio = float(dash_component.call("get_cooldown_ratio"))
	obs.append(can_dash_val)
	obs.append(dashing_val)
	obs.append(clamp(dash_cooldown_ratio, 0.0, 1.0))

	var level_norm = 0.0
	var xp_progress = 0.0
	if experience_manager:
		level_norm = clamp(float(experience_manager.current_level) / 50.0, 0.0, 1.0)
		if experience_manager.target_experience > 0:
			xp_progress = clamp(float(experience_manager.current_experience) / float(experience_manager.target_experience), 0.0, 1.0)
	obs.append(level_norm)
	obs.append(xp_progress)

	obs.append(_get_upgrade_ratio("sword_rate"))
	obs.append(_get_upgrade_ratio("sword_damage"))
	obs.append(_get_upgrade_ratio("axe_damage"))
	obs.append(_get_upgrade_ratio("player_speed"))

	var sword_cooldown_ratio = 0.0
	if sword_controller and sword_controller.has_node("Timer"):
		var sword_timer: Timer = sword_controller.get_node("Timer") as Timer
		if sword_timer and sword_timer.wait_time > 0.0:
			sword_cooldown_ratio = clamp(float(sword_timer.time_left) / float(sword_timer.wait_time), 0.0, 1.0)
	obs.append(sword_cooldown_ratio)

	var enemies = get_tree().get_nodes_in_group("enemy")
	var sorted_enemies: Array = []
	for enemy in enemies:
		if enemy is Node2D:
			var dist = player.global_position.distance_to(enemy.global_position)
			sorted_enemies.append({"enemy": enemy, "distance": dist})

	sorted_enemies.sort_custom(func(a, b): return a.distance < b.distance)

	var enemy_count_norm = clamp(float(enemies.size()) / 50.0, 0.0, 1.0)
	var mean_near_dist = 0.0
	if sorted_enemies.size() > 0:
		var sample_count = min(OBS_ENEMY_COUNT, sorted_enemies.size())
		for i in range(sample_count):
			mean_near_dist += float(sorted_enemies[i].distance)
		mean_near_dist /= float(sample_count)
	var mean_near_dist_norm = clamp(mean_near_dist / ARENA_DIAGONAL, 0.0, 1.0)
	obs.append(enemy_count_norm)
	obs.append(mean_near_dist_norm)

	var vial_rel_x = 0.0
	var vial_rel_y = 0.0
	var vial_dist_norm = 0.0
	var vials = get_tree().get_nodes_in_group("experience_vial")
	if vials.size() > 0:
		var best_dist = INF
		var best_vial: Node2D = null
		for vial in vials:
			if vial is Node2D:
				var vdist = player.global_position.distance_to(vial.global_position)
				if vdist < best_dist:
					best_dist = vdist
					best_vial = vial
		if best_vial:
			var rel_pos = (best_vial.global_position - player.global_position) / ARENA_SIZE
			vial_rel_x = clamp(rel_pos.x, -1.0, 1.0)
			vial_rel_y = clamp(rel_pos.y, -1.0, 1.0)
			vial_dist_norm = clamp(best_dist / ARENA_DIAGONAL, 0.0, 1.0)
	obs.append(vial_rel_x)
	obs.append(vial_rel_y)
	obs.append(vial_dist_norm)

	for i in range(OBS_ENEMY_COUNT):
		if i < sorted_enemies.size():
			var enemy = sorted_enemies[i].enemy
			var rel_pos = (enemy.global_position - player.global_position) / ARENA_SIZE
			obs.append(clamp(rel_pos.x, -1.0, 1.0))
			obs.append(clamp(rel_pos.y, -1.0, 1.0))

			var rel_vel = Vector2.ZERO
			if enemy.has_node("VelocityComponent"):
				rel_vel = enemy.get_node("VelocityComponent").velocity / 100.0
			obs.append(clamp(rel_vel.x, -1.0, 1.0))
			obs.append(clamp(rel_vel.y, -1.0, 1.0))

			obs.append(get_enemy_type_id(enemy))
		else:
			obs.append(0.0)
			obs.append(0.0)
			obs.append(0.0)
			obs.append(0.0)
			obs.append(0.0)

	return {"obs": obs}

func _get_upgrade_ratio(upgrade_id: String) -> float:
	if upgrade_manager == null:
		return 0.0
	if not upgrade_manager.current_upgrades.has(upgrade_id):
		return 0.0
	var entry = upgrade_manager.current_upgrades[upgrade_id]
	var quantity = float(entry.get("quantity", 0))
	var resource = entry.get("resource", null)
	if resource and resource.has_method("get"):
		var max_qty = float(resource.get("max_quantity"))
		if max_qty > 0.0:
			return clamp(quantity / max_qty, 0.0, 1.0)
	return clamp(quantity / 5.0, 0.0, 1.0)

func send_json(data: Dictionary) -> void:
	var json_str = JSON.stringify(data)
	var payload = json_str.to_utf8_buffer()
	var size_bytes = payload.size()
	var header = PackedByteArray()
	header.resize(4)
	header.encode_u32(0, size_bytes)
	client.put_data(header)
	client.put_data(payload)

func read_incoming_messages() -> void:
	var available = client.get_available_bytes()
	if available > 0:
		var result = client.get_data(available)
		if result[0] == OK:
			recv_buffer.append_array(result[1])
		else:
			return

	while recv_buffer.size() >= 4:
		var size = decode_u32_le(recv_buffer)
		if recv_buffer.size() < 4 + size:
			break
		var payload = recv_buffer.slice(4, 4 + size)
		recv_buffer = recv_buffer.slice(4 + size, recv_buffer.size())
		var text = payload.get_string_from_utf8()
		var data = JSON.parse_string(text)
		if data != null:
			handle_message(data)

func decode_u32_le(bytes: PackedByteArray) -> int:
	if bytes.size() < 4:
		return 0
	return int(bytes[0]) | (int(bytes[1]) << 8) | (int(bytes[2]) << 16) | (int(bytes[3]) << 24)

func simulate_action(action_name: String, pressed: bool) -> void:
	var ev = InputEventAction.new()
	ev.action = action_name
	ev.pressed = pressed
	ev.strength = 1.0 if pressed else 0.0
	Input.parse_input_event(ev)

func clear_actions() -> void:
	simulate_action("move_up", false)
	simulate_action("move_down", false)
	simulate_action("move_left", false)
	simulate_action("move_right", false)
	simulate_action("dash", false)

func get_player_health() -> float:
	if player and is_instance_valid(player) and player.has_node("HealthComponent"):
		return player.get_node("HealthComponent").current_health
	return 0.0

func get_player_health_percent() -> float:
	if player and is_instance_valid(player) and player.has_node("HealthComponent"):
		return player.get_node("HealthComponent").get_health_percent()
	return 0.0

func get_enemy_count() -> int:
	return get_tree().get_nodes_in_group("enemy").size()

func get_arena_difficulty() -> int:
	if arena_time_manager:
		return arena_time_manager.arena_difficulty
	return 0

func is_player_dead() -> bool:
	return get_player_health() <= 0.0

func is_episode_done() -> bool:
	if not player or not is_instance_valid(player):
		return true
	if is_player_dead():
		return true
	if arena_time_manager and arena_time_manager.has_method("is_finished"):
		return arena_time_manager.call("is_finished")
	return false

func is_moving() -> bool:
	if player and is_instance_valid(player) and player.has_node("VelocityComponent"):
		return player.get_node("VelocityComponent").velocity.length() > 0.1
	return false

func get_enemy_type_id(enemy: Node) -> float:
	var ename = enemy.name.to_lower()
	if ename.find("wizard") != -1:
		return 1.0
	if ename.find("bat") != -1:
		return 2.0
	return 0.0

func find_node_by_name(node_name: String) -> Node:
	return get_tree().root.find_child(node_name, true, false)

func find_node_by_class(class_name_arg: String) -> Node:
	var nodes = get_tree().root.find_children("", class_name_arg, true, false)
	if nodes.size() > 0:
		return nodes[0]
	return null

func get_port_from_args(default_port: int) -> int:
	var args = OS.get_cmdline_args()
	for arg in args:
		if arg.begins_with("--port="):
			return int(arg.split("=")[1])
	return default_port

func is_rl_run() -> bool:
	for arg in OS.get_cmdline_args():
		if arg == "--rl" or arg.begins_with("--port="):
			return true
	return false

func _get_top_action() -> String:
	var max_idx = 0
	var max_val = 0
	for i in range(action_histogram.size()):
		if action_histogram[i] > max_val:
			max_val = action_histogram[i]
			max_idx = i
	return ACTION_NAMES[max_idx]

func _create_hud() -> void:
	var canvas = CanvasLayer.new()
	canvas.name = "RLHud"
	canvas.layer = 100
	canvas.process_mode = Node.PROCESS_MODE_ALWAYS
	add_child(canvas)

	var panel = PanelContainer.new()
	panel.name = "Panel"
	var style = StyleBoxFlat.new()
	style.bg_color = Color(0.0, 0.0, 0.0, 0.75)
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)
	panel.position = Vector2(10, 10)
	canvas.add_child(panel)

	hud_label = Label.new()
	hud_label.name = "StatsLabel"
	hud_label.add_theme_font_size_override("font_size", 11)
	hud_label.add_theme_color_override("font_color", Color(0.0, 1.0, 0.4))
	hud_label.text = "RL Agent: Waiting..."
	panel.add_child(hud_label)

func _update_hud() -> void:
	if hud_label == null:
		return

	var pos_str = "N/A"
	var hp_str = "N/A"
	if player and is_instance_valid(player):
		pos_str = "(%d, %d)" % [int(player.global_position.x), int(player.global_position.y)]
		hp_str = "%.0f" % get_player_health()

	var action_name = ACTION_NAMES[last_action_id] if last_action_id < ACTION_NAMES.size() else "?"

	hud_label.text = "=== RL AGENT ===\n"
	hud_label.text += "Episode: %d  |  Best Reward: %.1f\n" % [episode_number, best_episode_reward]
	hud_label.text += "Step: %d  |  Reward: %.2f\n" % [episode_step_count, episode_total_reward]
	hud_label.text += "Action: %s (%d)  |  HP: %s\n" % [action_name, last_action_id, hp_str]
	hud_label.text += "Position: %s  |  Enemies: %d\n" % [pos_str, get_enemy_count()]
	hud_label.text += "Total Actions: %d" % total_actions_received
