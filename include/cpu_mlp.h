#ifndef CPU_MLP_H
#define CPU_MLP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "net.h"

class cpu_mlp: public net
{
public:
	cpu_mlp();
	~cpu_mlp();

	int load(const char* net_file);
	int run(const float* input, float* output);
	void kernel_free();
private:
	float *h_net;
	int *h_layer_dim;
	int total_item; //total item in the matrix
	int max_dim;
	int layer_num;
	
	float* h_inter_res[2];
	float non_linear(float x);	
};

#endif
