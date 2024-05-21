#include "random_process.h"

random_process::random_process()
{
	srand(time(0));
}

int random_process::get_rand(int num)
{
	return rand() % num;
}