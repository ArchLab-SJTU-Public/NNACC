#ifndef CL_MLP_H
#define CL_MLP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#define INTERACTIVE_MODE
//#define VISUAL

#define THREAD_PER_GROUP 32
#define TESTING_PLAT 0
#define TESTING_DEV 0

#include "net.h"


class cl_mlp : public net
{
public:
	cl_mlp();
	~cl_mlp();
	int load(const char* file_path);
	int run (const float* input, float* output);
	void kernel_free();
private:
	float *h_net;
	int *h_layer_dim;
	cl_uint max_dim;
	int total_item; //total item in the matrix
	int layer_num;

	float* h_inter_res[2];
	cl_mem d_inter_res[2];

	//cl related vars and methods
	cl_program load_program(cl_context context, const char* filename);
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_uint platnum, devnum;
	cl_context context;
	cl_command_queue queue;
	cl_mem *d_net;
	cl_program program;
	cl_kernel vmm;
	cl_event timer;
	//cl_ulong start, end, elipse;

	float non_linear(float x);
};

#endif
