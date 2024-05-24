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

const std::string action::get_name()
{
	switch (_feature)
	{
	case up: { return "上"; }break;
	case right: { return "右"; }break;
	case down: { return "下"; }break;
	case left: { return "左"; }break;
	case fixed: { return "原"; }break;
	default: {return "错误"; }break;
	}
}

void action::set_state_transfer(state* s, probability p)
{
	_state_transfer[s] = p;
}
