#include "test_data_set.h"

test_data_set::test_data_set() :
	_data{ torch::tensor({ 1,2,3,4,5 ,6,7,8,9,10}) },
	_target{ torch::tensor({10,9,8,7,6,5,4,3,2,1 }) }
{

}

torch::data::Example<> test_data_set::get(size_t index)
{
	return { _data[index], _target[index] };
}

torch::optional<size_t> test_data_set::size() const
{
	return 10;
}
