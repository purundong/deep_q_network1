#include "neural_network.h"

neural_network::neural_network(unsigned int hid_layer_size) :
	_in_layour{ register_module("_in_layou", torch::nn::Linear(torch::nn::LinearOptions(2,2))) },
	_hid_layer{ register_module("_hid_layer", torch::nn::Linear(torch::nn::LinearOptions(2,hid_layer_size))) },
	_out_layer{ register_module("_out_layer", torch::nn::Linear(torch::nn::LinearOptions(hid_layer_size,1))) }
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
