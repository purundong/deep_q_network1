#pragma once
#include "state.h"

class environment
{
	map_state_ptr _map_state;
public:
	environment(int row_size, int col_size, int trap, int target);//行数，列数，陷阱数，目标数

public:
	inline map_state_ptr get_map_state() { return _map_state; }
};

using environment_ptr = std::shared_ptr<environment>;
