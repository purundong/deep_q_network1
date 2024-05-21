#include "deep_q_network.h"
#include <QtWidgets/QApplication>
//#include "libtorch.h"
#include <iostream>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
	//std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	//std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	//std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;

	//torch::Device device(torch::kCUDA);
	//torch::Tensor tensor1 = torch::eye(3); // (A) tensor-cpu
	//torch::Tensor tensor2 = torch::eye(3, device); // (B) tensor-cuda
	//std::cout << tensor1 << std::endl;
	//std::cout << tensor2 << std::endl;
	deep_q_network w;
	w.show();
    return a.exec();
}
