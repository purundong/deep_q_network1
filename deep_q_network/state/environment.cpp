#include "environment.h"
#include "map_action_factory.h"

environment::environment(int row_size, int col_size, int trap, int target)
{
	for (int r = 0; r < row_size; ++r)
		for (int c = 0; c < col_size; ++c) {
			auto s = std::make_shared<state>(state_feature{ r, c });
			s->set_reward(0.0, 1.0);
			_map_state->emplace(s->get_name(), s);
		}

	auto factory = std::make_unique<map_action_factory>();
	for (int r = 1; r < row_size - 1; ++r)
		for (int c = 1; c < col_size - 1; ++c) {
			if (auto key = _map_state->find(state_feature::get_name(r, c)); key != _map_state->end())
				key->second->set_map_action(factory->create_map_action(*_map_state, r, c));
		}

	factory = std::make_unique<map_action_top_factory>();
	for (int c = 1; c < col_size - 1; ++c)
	{	//第一行
		if (auto key = _map_state->find(state_feature::get_name(0, c)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, 0, c));
	}

	factory = std::make_unique<map_action_bot_factory>();
	for (int c = 1; c < col_size - 1; ++c)
	{	//最后一行
		if (auto key = _map_state->find(state_feature::get_name(row_size - 1, c)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, row_size - 1, c));
	}

	factory = std::make_unique<map_action_left_factory>();
	for (int r = 1; r < row_size - 1; ++r)
	{//第一列
		if (auto key = _map_state->find(state_feature::get_name(r, 0)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, r, 0));
	}

	factory = std::make_unique<map_action_right_factory>();
	for (int r = 1; r < row_size - 1; ++r)
	{//最后一列
		if (auto key = _map_state->find(state_feature::get_name(r, col_size - 1)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, r, col_size - 1));
	}

	factory = std::make_unique<map_action_upper_left_factory>();

	if (auto key = _map_state->find(state_feature::get_name(0, 0)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, 0, 0));

	factory = std::make_unique<map_action_lower_left_factory>();

	if (auto key = _map_state->find(state_feature::get_name(row_size - 1, 0)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, row_size - 1, 0));

	factory = std::make_unique<map_action_upper_right_factory>();

	if (auto key = _map_state->find(state_feature::get_name(0, col_size - 1)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, 0, col_size - 1));

	factory = std::make_unique<map_action_lower_right_factory>();

	if (auto key = _map_state->find(state_feature::get_name(row_size - 1, col_size - 1)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, row_size - 1, col_size - 1));

}
