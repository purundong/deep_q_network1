#include "environment.h"
#include "map_action_factory.h"
#include "random_process.h"

environment::environment(int row_size, int col_size, int trap, int target) :
	_map_state{ std::make_shared<map_state>() }
{
	for (int r = 1; r <= row_size; ++r)
		for (int c = 1; c <= col_size; ++c) {
			auto s = std::make_shared<state>(state_feature{ r, c });
			s->set_reward(0.0, 1.0);
			_map_state->emplace(s->get_name(), s);
		}

	auto factory = std::make_unique<map_action_factory>();
	for (int r = 2; r <= row_size - 1; ++r)
		for (int c = 2; c < col_size - 1; ++c) {
			if (auto key = _map_state->find(state_feature::get_name(r, c)); key != _map_state->end())
				key->second->set_map_action(factory->create_map_action(*_map_state, r, c));
		}

	factory = std::make_unique<map_action_top_factory>();
	for (int c = 1; c <= col_size - 1; ++c)
	{	//第一行
		if (auto key = _map_state->find(state_feature::get_name(0, c)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, 0, c));
	}

	factory = std::make_unique<map_action_bot_factory>();
	for (int c = 2; c <= col_size - 1; ++c)
	{	//最后一行
		if (auto key = _map_state->find(state_feature::get_name(row_size - 1, c)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, row_size - 1, c));
	}

	factory = std::make_unique<map_action_left_factory>();
	for (int r = 2; r <= row_size - 1; ++r)
	{//第一列
		if (auto key = _map_state->find(state_feature::get_name(r, 0)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, r, 0));
	}

	factory = std::make_unique<map_action_right_factory>();
	for (int r = 2; r <= row_size - 1; ++r)
	{//最后一列
		if (auto key = _map_state->find(state_feature::get_name(r, col_size - 1)); key != _map_state->end())
			key->second->set_map_action(factory->create_map_action(*_map_state, r, col_size - 1));
	}

	factory = std::make_unique<map_action_upper_left_factory>();

	if (auto key = _map_state->find(state_feature::get_name(1, 1)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, 1, 1));

	factory = std::make_unique<map_action_lower_left_factory>();

	if (auto key = _map_state->find(state_feature::get_name(row_size, 1)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, row_size, 1));

	factory = std::make_unique<map_action_upper_right_factory>();

	if (auto key = _map_state->find(state_feature::get_name(1, col_size)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, 1, col_size));

	factory = std::make_unique<map_action_lower_right_factory>();

	if (auto key = _map_state->find(state_feature::get_name(row_size, col_size)); key != _map_state->end())
		key->second->set_map_action(factory->create_map_action(*_map_state, row_size, col_size));

	random_process rand;
	std::unordered_set<std::string> num_set;
	for (int i = 0; i < trap; ++i) {
		while (true)
		{
			int row = rand.get_rand(row_size) + 1;
			int col = rand.get_rand(col_size) + 1;
			auto name = state_feature::get_name(row, col);
			if (num_set.find(name) == num_set.end())
			{
				if (auto key = _map_state->find(name); key != _map_state->end())
					key->second->reset_reward(-1, 1);
				num_set.emplace(name);
				break;
			}
		}
	}

	for (int i = 0; i < target; ++i) {
		while (true)
		{
			int row = rand.get_rand(row_size) + 1;
			int col = rand.get_rand(col_size) + 1;
			auto name = state_feature::get_name(row, col);
			if (num_set.find(name) == num_set.end())
			{
				if (auto key = _map_state->find(name); key != _map_state->end())
					key->second->reset_reward(1, 1);
				num_set.emplace(name);
				break;
			}
		}
	}
}

trajectory_ptr environment::sampling(int step_count)
{
	auto trajectory_obj = std::make_shared<trajectory>();
	auto curr_state_obj = _map_state->begin()->second.get();
	for (int i = 0; i < step_count; ++i) {
		auto action_obj = curr_state_obj->sample_action().get();
		auto next_state_obj = action_obj->sample_state();
		auto reword = next_state_obj->sample_reword();
		trajectory_obj->push_sample({ curr_state_obj, next_state_obj ,action_obj, reword });
		curr_state_obj = next_state_obj;
	}
	trajectory_obj->init_iter();
	return trajectory_obj;
}

void environment::update_agent(neural_network_ptr network, torch::Device dev_type)
{
	for (auto& [state_name, state_obj] : *_map_state) {
		for (auto& [action_name, action_obj] : *state_obj->get_actions()) {
			auto& state_feature_obj = state_obj->get_feature();
			auto feature = torch::empty({ 1 , 3 }, dev_type);
			feature[0][0] = (float)state_feature_obj._x;
			feature[0][1] = (float)state_feature_obj._y;
			feature[0][2] = (float)action_obj->get_feature();
			auto y = network->forward(feature);
			action_obj->set_value(y.item<double>());
		}
		state_obj->update_policy_greedy();
		state_obj->update_value();
	}
}

void environment::update_state_policy_value()
{
	for (auto& [state_name, state_obj] : *_map_state) {
		state_obj->update_policy_greedy();
		state_obj->update_value();
	}
}

replay_buf::replay_buf(const std::list<std::list<sample>::iterator>& batch, neural_network_ptr target_network, float gama, torch::Device dev_type) :
	_buf_size{ batch.size() }
{
	_target = torch::empty({ (long long)batch.size() , 1 }, dev_type);
	_feature = torch::empty({ (long long)batch.size() , 3 }, dev_type);
	for (int i = 0; auto & sampling : batch) {
		auto& curr_state_feature_obj = sampling->_next_state->get_feature();

		_feature[i][0] = (float)curr_state_feature_obj._x;
		_feature[i][1] = (float)curr_state_feature_obj._y;
		_feature[i][2] = (float)sampling->_curr_action->get_feature();

		auto x = torch::empty({ 5 , 3 }, dev_type);
		auto& next_state_feature_obj = sampling->_next_state->get_feature();

		x[0][0] = (float)next_state_feature_obj._x;
		x[0][1] = (float)next_state_feature_obj._y;
		x[0][2] = (float)action::feature::up;

		x[1][0] = (float)next_state_feature_obj._x;
		x[1][1] = (float)next_state_feature_obj._y;
		x[1][2] = (float)action::feature::right;

		x[2][0] = (float)next_state_feature_obj._x;
		x[2][1] = (float)next_state_feature_obj._y;
		x[2][2] = (float)action::feature::down;

		x[3][0] = (float)next_state_feature_obj._x;
		x[3][1] = (float)next_state_feature_obj._y;
		x[3][2] = (float)action::feature::left;

		x[4][0] = (float)next_state_feature_obj._x;
		x[4][1] = (float)next_state_feature_obj._y;
		x[4][2] = (float)action::feature::fixed;

		_target[i][0] = sampling->_reword + gama * torch::max(target_network->forward(x));
		++i;
	}
}

replay_buf::replay_buf(const std::list<sample>& samples, torch::Device dev_type) :
	_buf_size{ samples.size() }
{
	_target = torch::empty({ (long long)samples.size() , 16 }, dev_type);
	_feature = torch::empty({ (long long)samples.size() , 3 }, dev_type);

	for (int i = 0; auto & sampling : samples) {
		auto& curr_state_feature_obj = sampling._next_state->get_feature();

		_feature[i][0] = (float)curr_state_feature_obj._x;
		_feature[i][1] = (float)curr_state_feature_obj._y;
		_feature[i][2] = (float)sampling._curr_action->get_feature();
		auto& next_state_feature_obj = sampling._next_state->get_feature();
		//_target[i][0] = sampling->_reword + gama * torch::max(target_network->forward(x));
		_target[i][0] = sampling._reword;
		_target[i][1] = (float)next_state_feature_obj._x;
		_target[i][2] = (float)next_state_feature_obj._y;
		_target[i][3] = (float)action::feature::up;

		_target[i][4] = (float)next_state_feature_obj._x;
		_target[i][5] = (float)next_state_feature_obj._y;
		_target[i][6] = (float)action::feature::right;

		_target[i][7] = (float)next_state_feature_obj._x;
		_target[i][8] = (float)next_state_feature_obj._y;
		_target[i][9] = (float)action::feature::down;

		_target[i][10] = (float)next_state_feature_obj._x;
		_target[i][11] = (float)next_state_feature_obj._y;
		_target[i][12] = (float)action::feature::left;

		_target[i][13] = (float)next_state_feature_obj._x;
		_target[i][14] = (float)next_state_feature_obj._y;
		_target[i][15] = (float)action::feature::fixed;

		++i;
	}

}

torch::data::Example<> replay_buf::get(size_t index)
{
	return { _feature[index], _target[index] };
}

torch::optional<size_t> replay_buf::size() const
{
	return _buf_size;
}

std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::SequentialSampler>> trajectory::get_replay_buf(neural_network_ptr target_network, size_t model_num, float gama, torch::Device dev_type)
{
	std::list<std::list<sample>::iterator> batch;
	for (int i = 0; i < model_num; ++i) {
		batch.push_back(_iter++);
		if (_iter == _samples.end())
			_iter = _samples.begin();
	}
	return torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(replay_buf(batch, target_network, gama, dev_type), 1);
}

std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::RandomSampler>> trajectory::get_random_replay_buf(neural_network_ptr target_network, size_t model_num, float gama, torch::Device dev_type)
{
	std::list<std::list<sample>::iterator> batch;
	for (int i = 0; i < model_num; ++i) {
		batch.push_back(_iter++);
		if (_iter == _samples.end())
			_iter = _samples.begin();
	}
	return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(replay_buf(batch, target_network, gama, dev_type), 1);
}

std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::RandomSampler>> trajectory::get_random_replay_buf(torch::Device dev_type)
{
	return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(replay_buf(_samples, dev_type), 1);
}

void trajectory::push_sample(sample&& sample_obj)
{
	_samples.push_back(std::move(sample_obj));
}

void trajectory::init_iter()
{
	_iter = _samples.begin();
}
