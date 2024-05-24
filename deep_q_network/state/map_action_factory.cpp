#include "map_action_factory.h"
#include "state.h"

map_action_ptr map_action_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = std::make_shared<map_action>();
	(*map_action_obj)[action::feature::up] = std::make_shared<action>(action::feature::up);
	auto name = state_feature::get_name(x - 1, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::up]->set_state_transfer(s_key->second.get(), 1.0);

	(*map_action_obj)[action::feature::right] = std::make_shared<action>(action::feature::right);
	name = state_feature::get_name(x, y + 1);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::right]->set_state_transfer(s_key->second.get(), 1.0);

	(*map_action_obj)[action::feature::down] = std::make_shared<action>(action::feature::down);
	name = state_feature::get_name(x + 1, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::down]->set_state_transfer(s_key->second.get(), 1.0);

	(*map_action_obj)[action::feature::left] = std::make_shared<action>(action::feature::left);
	name = state_feature::get_name(x, y - 1);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::left]->set_state_transfer(s_key->second.get(), 1.0);

	(*map_action_obj)[action::feature::fixed] = std::make_shared<action>(action::feature::fixed);
	name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::fixed]->set_state_transfer(s_key->second.get(), 1.0);

	return map_action_obj;
}

map_action_ptr map_action_top_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::up]->set_state_transfer(s_key->second.get(), 1.0);
	return map_action_obj;
}

map_action_ptr map_action_bot_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::down]->set_state_transfer(s_key->second.get(), 1.0);
	return map_action_obj;
}

map_action_ptr map_action_left_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::left]->set_state_transfer(s_key->second.get(), 1.0);
	return map_action_obj;
}

map_action_ptr map_action_right_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end())
		(*map_action_obj)[action::feature::right]->set_state_transfer(s_key->second.get(), 1.0);
	return map_action_obj;
}

map_action_ptr map_action_upper_left_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end()) {
		(*map_action_obj)[action::feature::up]->set_state_transfer(s_key->second.get(), 1.0);
		(*map_action_obj)[action::feature::left]->set_state_transfer(s_key->second.get(), 1.0);
	}
	return map_action_obj;
}

map_action_ptr map_action_lower_left_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end()) {
		(*map_action_obj)[action::feature::down]->set_state_transfer(s_key->second.get(), 1.0);
		(*map_action_obj)[action::feature::left]->set_state_transfer(s_key->second.get(), 1.0);
	}
	return map_action_obj;
}

map_action_ptr map_action_upper_right_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end()) {
		(*map_action_obj)[action::feature::up]->set_state_transfer(s_key->second.get(), 1.0);
		(*map_action_obj)[action::feature::right]->set_state_transfer(s_key->second.get(), 1.0);
	}
	return map_action_obj;
}

map_action_ptr map_action_lower_right_factory::create_map_action(std::map<std::string, state_ptr>& states, int x, int y)
{
	auto map_action_obj = map_action_factory::create_map_action(states, x, y);
	auto name = state_feature::get_name(x, y);
	if (auto s_key = states.find(name); s_key != states.end()) {
		(*map_action_obj)[action::feature::down]->set_state_transfer(s_key->second.get(), 1.0);
		(*map_action_obj)[action::feature::right]->set_state_transfer(s_key->second.get(), 1.0);
	}
	return map_action_obj;
}
