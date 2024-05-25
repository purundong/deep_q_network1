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

trajectory_ptr environment::sampling(int step_count)
{
	auto trajectory_obj = std::make_shared<trajectory>();
	auto curr_state_obj = _map_state->begin()->second;
	for (int i = 0; i < step_count; ++i) {
		auto action_obj = curr_state_obj->sample_action();
		auto next_state_obj = action_obj->sample_state();
		auto reword = next_state_obj->sample_reword();
		trajectory_obj->push_back({ curr_state_obj ,action_obj,reword });
	}
	return trajectory_obj;
}

replay_buf::replay_buf(trajectory_ptr trajectory_obj, neural_network_ptr target_network, torch::DeviceType dev_type)
{
	at::TensorOptions options;
	options.device(dev_type);
	options.dtype(torch::kFloat32);
	_target = torch::empty({ (long long)trajectory_obj->size() , 1 }, options);
	_feature = torch::empty({ (long long)trajectory_obj->size() , 3 }, options);
	for (int i = 0; auto & sampling : *trajectory_obj) {
		auto x = torch::empty({ 1 , 3 }, options);
		auto& state_feature_obj = sampling._state->get_feature();
		x[0] = (float)state_feature_obj._x;
		x[1] = (float)state_feature_obj._y;
		x[2] = (float)sampling._action->get_feature();
		_target[i][0] = sampling._reword + target_network->forward(x);
		_feature[i] = x;
		++i;
	}
}

torch::data::Example<> replay_buf::get(size_t index)
{
	return torch::data::Example<>();
}

torch::optional<size_t> replay_buf::size() const
{
	return torch::optional<size_t>();
}
