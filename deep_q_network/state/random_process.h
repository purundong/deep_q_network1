#pragma once
#include <vector>
#include <random>

class random_process
{
public:
	random_process();
	int get_rand(int num);
public:
	template<typename T>
	static int sampling(const T& probability);
};

template<typename T>
inline int random_process::sampling(const T& probability)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dis(probability.begin(), probability.end());
	return dis(gen);
}
