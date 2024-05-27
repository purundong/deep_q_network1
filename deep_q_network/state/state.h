#pragma once
#include <string>
#include "declare.h"
#include <map>
#include "action.h"
#include <format>
#include <QDialog>

struct state_feature
{
	const int _x, _y;
public:
	inline static std::string get_name(const int x, const int y) {
		return std::format("{:0>3}_{:0>3}", x, y);
	}

public:
	inline std::string get_name() {
		return state_feature::get_name(_x, _y);
	}
};

class state;
using map_state = std::map<std::string, state_ptr>;
using map_state_ptr = std::shared_ptr<map_state>;

class state : public std::enable_shared_from_this<state>
{
private:
	state_feature _feature;
	double _value;//当前状态的价值
private:
	std::map<reward, probability> _rewards;
	map_action_ptr _map_action; //当前状态的动作
	std::map<action::action::feature, probability> _policy; //当前状态的动作概率(策略)
public:
	inline const state_feature& get_feature() { return _feature; }
	inline double value() { return _value; };
	inline void set_value(const double& value) { _value = value; };
	inline probability policy(const action::feature& type) { return _policy[type]; }
	inline std::map<action::action::feature, probability>& policy() { return _policy; }
	inline map_action_ptr get_actions() {return _map_action;}; 
	inline std::map<reward, probability>& get_rewards() { return _rewards; }
	inline void set_reward(reward r, probability p) { _rewards[r] = p; }
	inline void reset_reward(reward r, probability p) { _rewards.clear(); _rewards[r] = p; }
	inline void set_map_action(map_action_ptr actions) { _map_action = actions; }
public:
	double reward_expectations(); //获取当前状态的奖励期望
	action_ptr sample_action(); //根据策略抽样一个动作
	reward sample_reword(); //采样一个奖励
	void update_policy_greedy();
	void update_value();
	double max_max_action_value();
public:
	state(const state_feature& feature);
	state(state_feature&& feature);
	std::string get_name();
	std::shared_ptr<QDialog> make_set_reword(); //获取当前状态的控件
	std::shared_ptr<QDialog> make_set_state_transfers(map_state_ptr states); //获取当前状态的控件
	std::shared_ptr<QDialog> make_show_state_info();
};

