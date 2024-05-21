#include "action.h"
#include <vector>
#include <algorithm>
#include "random_process.h"

state_ptr action::sample_state()
{
	std::vector<double> vec(_state_transfer.size());
	std::vector<state_ptr> states;
	states.reserve(_state_transfer.size());
	auto fun = [&states](const std::pair<std::weak_ptr<state>, probability>& pa) {
		states.push_back(pa.first.lock());
		return pa.second;
		};
	std::transform(_state_transfer.begin(), _state_transfer.end(), vec.begin(), fun);
	auto a = random_process::sampling(vec);
	return states[a];
}
