#include "state.h"
#include "declare.h"
#include <vector>
#include <algorithm>
#include <format>
#include "random_process.h"
#include "ui_reword.h"
#include "ui_state_transfers.h"
#include "ui_state_info.h"
#include <iostream>

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
	
	auto a = (action::action::feature)(random_process::sampling(probability_vec) + 1);
	if (_map_action == nullptr) {
		std::cout << get_name() << "\n";
		this;
	}
	auto& action = (*_map_action)[a];
	return action;
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

void state::update_policy_greedy()
{
	auto max = std::max_element(_map_action->begin(), _map_action->end(), [](const auto& a, const auto& b) {return a.second->value() < b.second->value(); });
	for (auto& [action_type, action_obj] : *_map_action)
		_policy[action_type] = (action_type == max->first) ? 1.0 : 0.0;
}

void state::update_value()
{
	_value = 0.0;
	for (auto& [action_type, action_obj] : *_map_action) {
		_value += _policy[action_type] * action_obj->value();
	}
}

double state::max_max_action_value()
{
	auto fun = [](const auto& a, const auto& b) {return a.second->value() < b.second->value(); };
	auto max = std::max_element(_map_action->begin(), _map_action->end(), fun);
	return max->second->value();
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

std::string state::get_name()
{
	return _feature.get_name();
}

std::shared_ptr<QDialog> state::make_set_reword()
{
	return std::dynamic_pointer_cast<QDialog>(std::make_shared<ui_reword>(shared_from_this()));
}

std::shared_ptr<QDialog> state::make_set_state_transfers(map_state_ptr states)
{
	return std::dynamic_pointer_cast<QDialog>(std::make_shared<ui_state_transfers>(shared_from_this(), states));
}

std::shared_ptr<QDialog> state::make_show_state_info()
{
	return std::dynamic_pointer_cast<QDialog>(std::make_shared<ui_state_info>(shared_from_this()));
}
