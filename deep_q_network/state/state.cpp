#include "state.h"
#include "declare.h"
#include <vector>
#include <algorithm>
#include "random_process.h"

double state::reward_expectations()
{
	double expectations = 0.0;
	std::for_each(_rewards.begin(), _rewards.end(), [&expectations](auto& p) {expectations += p.first * p.second; });
	return expectations;
}

action_ptr state::sample_action()
{
	std::vector<double> probability_vec(_policy.size());
	auto fun = [](const std::pair<const action::action::feature, double>& pa) { return pa.second; };
	std::transform(_policy.begin(), _policy.end(), probability_vec.begin(), fun);
	auto a = (action::action::feature)random_process::sampling(probability_vec);
	return (*_map_action)[a];
}

reward state::sample_reword()
{
	std::vector<double> probability_vec(_rewards.size());
	std::vector<double> rewards;
	rewards.reserve(_rewards.size());
	auto fun = [&rewards](const std::pair<const reward, probability>& pa) {
		rewards.push_back(pa.first);
		return pa.second;
		};
	std::transform(_rewards.begin(), _rewards.end(), probability_vec.begin(), fun);
	auto a = random_process::sampling(probability_vec);
	return rewards[a];
}

state::state(const state_feature& feature) :
	_feature{ feature },
	_policy({ {action::feature::up, 0.25},
		{action::feature::right, 0.25},
		{action::feature::down, 0.25},
		{action::feature::left, 0.25},
		{action::feature::fixed, 0.25} }),
	_value{ 0 }
{
}

state::state(state_feature&& feature) :
	_feature{ std::move(feature) },
	_policy({ {action::feature::up, 0.25},
		{action::feature::right, 0.25},
		{action::feature::down, 0.25},
		{action::feature::left, 0.25},
		{action::feature::fixed, 0.25} }),
	_value{ 0 }
{
}
