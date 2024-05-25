#pragma once
#include "state.h"
#include "neural_network.h"

struct sample
{
	state_ptr _state;
	action_ptr _action;
	double _reword;
};

using trajectory = std::list<sample>;
using trajectory_ptr = std::shared_ptr<trajectory>;

class environment
{
	map_state_ptr _map_state;
public:
	environment(int row_size, int col_size, int trap, int target);//行数，列数，陷阱数，目标数
	trajectory_ptr sampling(int step_count);
public:
	inline map_state_ptr get_map_state() { return _map_state; }
};

using environment_ptr = std::shared_ptr<environment>;

class replay_buf :public torch::data::Dataset<replay_buf>
{
	at::Tensor _target, _feature;
public:
	replay_buf(trajectory_ptr trajectory_obj, neural_network_ptr target_network, torch::DeviceType dev_type);
public:
	virtual torch::data::Example<> get(size_t index);
	virtual torch::optional<size_t> size() const;
};