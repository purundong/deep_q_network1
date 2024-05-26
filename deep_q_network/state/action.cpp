#include "action.h"
#include <vector>
#include <algorithm>
#include "state.h"
#include "random_process.h"

state* action::sample_state()
{
	std::vector<double> vec(_state_transfer.size());
	std::vector<state*> states;
	states.reserve(_state_transfer.size());
	auto fun = [&states](const std::pair<state*, probability>& pa) {
		states.push_back(pa.first);
		return pa.second;
		};
	std::transform(_state_transfer.begin(), _state_transfer.end(), vec.begin(), fun);
	auto a = random_process::sampling(vec);
	return states[a];
}

void action::set_state_transfer(state* s, probability p)
{
	_state_transfer[s] = p;
}
