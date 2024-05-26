#include "neural_network.h"

neural_network::neural_network(unsigned int hid_layer_size) :
	_in_layour{ register_module("_in_layou", torch::nn::Linear(torch::nn::LinearOptions(3,3))) },
	_hid_layer{ register_module("_hid_layer", torch::nn::Linear(torch::nn::LinearOptions(3, hid_layer_size))) },
	_out_layer{ register_module("_out_layer", torch::nn::Linear(torch::nn::LinearOptions(hid_layer_size, 1))) }
{
}

neural_network::~neural_network()
{
}

torch::Tensor neural_network::forward(torch::Tensor x)
{
	x = torch::nn::functional::relu(_in_layour->forward(x));
	x = torch::nn::functional::relu(_hid_layer->forward(x));
	return _out_layer->forward(x);
}

void neural_network::show()
{
	auto params = parameters();
	for (int i = 0; i < params.size(); ++i)
		std::cout << params[i] << "\n---------\n";
}

void neural_network::copy_params(const neural_network& network)
{
	auto source = network.parameters();
	auto target = parameters();
	for (int i = 0; i < target.size(); ++i)
		target[i] = source[i].clone();
}
