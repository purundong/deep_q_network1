#pragma once
#include "state.h"
class environment
{
	std::map<std::string, state_ptr> _map_state;
public:
	environment(int row_size, int col_size, int trap, int target);//行数，列数，陷阱数，目标数

};

