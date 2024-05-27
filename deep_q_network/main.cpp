#include "deep_q_network.h"
#include <QtWidgets/QApplication>
#include <iostream>
#include "libtorch.h"
#include "neural_network.h"
#include "test_data_set.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	deep_q_network w;
	w.show();
	//std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	//std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	//std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;
	try
	{
		//auto buf_data = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(test_data_set(), 1);
		//for (auto& buf_key : *buf_data)
		//{
		//	std::cout << buf_key[0].data;
		//	std::cout << "\n------\n";
		//	std::cout << buf_key[0].target;
		//}
		//auto ss1 = torch::tensor({ { 0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 } });
		//
		//std::cout << ss1[0].slice(0, 1, 16).view({5,3}) << "\n-----\n";

	}
	catch (const std::runtime_error& e) {
		auto str = e.what();
		return -1;
	}
	catch (const at::Error& e)
	{
		auto str = e.msg();
	}

	//auto datas = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(test_data_set(), 3);
	//for (auto& data_key : *datas) {
	//	for (int i = 0; i < data_key.size(); ++i)
	//	{
	//		std::cout << data_key[i].data<<"data";
	//		std::cout << "\n------\n";
	//		std::cout << data_key[i].target<<"target";
	//		std::cout << "\n!!!!!!!!!\n";
	//		
	//	}
	//}
	return a.exec();
}