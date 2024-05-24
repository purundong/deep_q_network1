#pragma once
#include "libtorch.h"

class neural_network :public torch::nn::Module
{
	torch::nn::Linear _in_layour, _hid_layer, _out_layer;
public:
	neural_network(unsigned int hid_layer_size);
	virtual ~neural_network();
public:
	torch::Tensor forward(torch::Tensor x);
};

