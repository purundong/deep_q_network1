#include "neural_network.h"

neural_network::neural_network(unsigned int hid_layer_size) :
	_in_layour{ register_module("_in_layou", torch::nn::Linear(torch::nn::LinearOptions(3,3))) },
	//隐藏层 三个输入 分别是(状态的x与y,和动作的特征值,输出100个)
	_hid_layer{ register_module("_hid_layer", torch::nn::Linear(torch::nn::LinearOptions(3, hid_layer_size))) },
	//输出层hid_layer_size(100)个输入1个输出
	_out_layer{ register_module("_out_layer", torch::nn::Linear(torch::nn::LinearOptions(hid_layer_size, 1))) }
{
	torch::nn::init::kaiming_uniform_(_in_layour->weight);
	torch::nn::init::kaiming_uniform_(_hid_layer->weight);
}

neural_network::~neural_network()
{
}

torch::Tensor neural_network::forward(torch::Tensor x)
{
	x = torch::nn::functional::relu(_in_layour->forward(x));
	//x = _in_layour->forward(x);
	x = torch::nn::functional::relu(_hid_layer->forward(x));//relu作为隐藏层的激活函数
	return _out_layer->forward(x);
}

void neural_network::show()
{
	auto params = parameters();
	//eval();
	//train()
	for (int i = 0; i < params.size(); ++i)
		std::cout << params[i] << "\n---------\n";
	for (int i = 0; i < params.size(); ++i)
		std::cout << params[i] << "\n---------\n";
}

void neural_network::show_grad()
{
	auto grads = parameters();
	for (int i = 0; i < grads.size(); ++i) {
		std::cout << grads[i].grad() << "\n---------\n";
	}
}

void neural_network::copy_params(const neural_network& network)
{
	auto source = network.parameters();
	auto target = parameters();
	for (int i = 0; i < target.size(); ++i)
		target[i] = source[i].clone();
}
