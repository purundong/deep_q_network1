#pragma once
#include "action.h"

class map_action_factory
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_top_factory :public map_action_factory //上边
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_bot_factory :public map_action_factory //下边
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_left_factory :public map_action_factory //左边
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_right_factory :public map_action_factory //右边
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_upper_left_factory :public map_action_factory //左上角
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_lower_left_factory :public map_action_factory //左下角
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_upper_right_factory :public map_action_factory //右上角
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

class map_action_lower_right_factory :public map_action_factory //右下角
{
public:
	virtual map_action_ptr create_map_action(std::map<std::string, state_ptr>& states, int x, int y);
};

