#pragma once
#include "libtorch.h"
#include <memory>

class neural_network :public torch::nn::Module
{
	torch::nn::Linear _in_layour, _hid_layer, _out_layer;
public:
	neural_network(unsigned int hid_layer_size);
	virtual ~neural_network();
public:
	torch::Tensor forward(torch::Tensor x);
	void show();
	void show_grad();
	void copy_params(const neural_network& network);
};
using neural_network_ptr = std::shared_ptr<neural_network>;

