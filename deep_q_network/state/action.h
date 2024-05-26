#pragma once
#include "declare.h"
#include <memory>
#include <map>
#include <string>

class state;
using state_ptr = std::shared_ptr<state>;

class action
{
public:
	enum feature : int { up = 1, right = 2, down = 3, left = 4, fixed = 5 };
	static const int action_size;
public:
	const feature _feature;
	double _value; //动作的价值
	std::map<state*, probability> _state_transfer;
public:
	inline const std::string get_name() { return action::get_name(_feature); }
	inline double value() { return _value; };
	inline void set_value(const double& value) { _value = value; };
	inline const action::feature& get_feature() { return _feature; }
	inline std::map<state*, probability>& get_state_transfer() {return _state_transfer;}
public:
	state* sample_state(); //根据动作转移概率抽样一个状态
	
	void set_state_transfer(state* s, probability p);
public:
	inline static const std::string get_name(action::feature feat) {
		switch (feat)
		{
		case up: { return "上"; }break;
		case right: { return "右"; }break;
		case down: { return "下"; }break;
		case left: { return "左"; }break;
		case fixed: { return "原"; }break;
		default: {return "错误"; }break;
		}
	}
};
using action_ptr = std::shared_ptr<action>;
using map_action = std::map<action::action::feature, action_ptr>;
using map_action_ptr = std::shared_ptr<map_action>;
