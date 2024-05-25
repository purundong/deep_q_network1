#include "deep_q_network.h"
#include <QtWidgets/QApplication>
#include <iostream>
#include "libtorch.h"
#include "neural_network.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
	//std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	//std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	//std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;

	torch::Device device(torch::kCUDA);
	//neural_network network(3);
	//network.to(device);
	//network.show();
	//torch::Tensor tensor1 = torch::empty({3,4}, device); // (A) tensor-cpu
	//torch::Tensor tensor2 = torch::empty({1,4}, device); // (B) tensor-cuda
	//tensor2[0][0] = 1.0;
	//tensor2[0][1] = 2.0;
	//tensor2[0][2] = 3.0;
	//tensor2[0][3] = 3.0;
	//tensor1[1] = tensor2.view({4});
	//std::cout << tensor1 << std::endl;
	//std::cout << tensor2 << std::endl;
	//deep_q_network w;
	//w.show();
    return a.exec();
}