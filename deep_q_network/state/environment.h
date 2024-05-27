#pragma once
#include "state.h"
#include "neural_network.h"

class trajectory;
using trajectory_ptr = std::shared_ptr<trajectory>;

struct sample
{
	state* _curr_state, *_next_state;
	action* _curr_action;
	double _reword;
};

class replay_buf : public torch::data::Dataset<replay_buf>
{
	at::Tensor _target, _feature;
	size_t _buf_size;
public:
	replay_buf(const std::list<std::list<sample>::iterator>& batch, neural_network_ptr target_network, float gama, torch::Device dev_type);
	replay_buf(const std::list<sample>& samples, torch::Device dev_type);
public:
	virtual torch::data::Example<> get(size_t index);
	virtual torch::optional<size_t> size() const;
};

class trajectory : public std::enable_shared_from_this<trajectory>
{
	std::list<sample> _samples;
	std::list<sample>::iterator _iter;
public:
	std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::SequentialSampler>> get_replay_buf(neural_network_ptr target_network, size_t model_num, float gama, torch::Device dev_type);
	std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::RandomSampler>> get_random_replay_buf(neural_network_ptr target_network, size_t model_num, float gama, torch::Device dev_type);
	std::unique_ptr<torch::data::StatelessDataLoader<replay_buf, torch::data::samplers::RandomSampler>> get_random_replay_buf(torch::Device dev_type);
public:
	void push_sample(sample&& sample_obj);
	void init_iter();
public:
	inline std::list<sample>& get_samples() {return _samples;}
};

class environment
{
	map_state_ptr _map_state;
public:
	environment(int row_size, int col_size, int trap, int target);//行数，列数，陷阱数，目标数
	trajectory_ptr sampling(int step_count);
	void update_agent(neural_network_ptr network, torch::Device dev_type);
	void update_state_policy_value();
public:
	inline map_state_ptr get_map_state() { return _map_state; }
	inline bool empty() {return _map_state->empty();}
};

using environment_ptr = std::shared_ptr<environment>;