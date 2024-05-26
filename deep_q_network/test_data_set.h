#pragma once
#include "libtorch.h"

class test_data_set : public torch::data::Dataset<test_data_set>
{
	torch::Tensor _data, _target;
public:
	test_data_set();
public:
	virtual torch::data::Example<> get(size_t index);
	virtual torch::optional<size_t> size() const;
};

