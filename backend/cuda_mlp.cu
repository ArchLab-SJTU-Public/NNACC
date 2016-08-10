#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_mlp.h>

using namespace std;

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__device__ float sigmoid(float x) {
	return 1.0 / (1 + exp(0 - x));
}

__global__ void MatMulKernel(Matrix weight, float *input) {
	int input_size = weight.height;
	int output_size = weight.width;

	int tid = threadIdx.x;
	// load input into shared memory.
	//__shared__ float share_input[1024];
	extern __shared__ float share_input[];
	for (int i = tid; i < input_size - 1; i += blockDim.x) {
		share_input[i] = input[i];
	}
	__syncthreads();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < output_size) {
		float sum = 0;
		for (int i = 0; i < input_size; i++) {
			float* row = weight.elements + i * output_size;
			if (i != input_size - 1) {
				sum += share_input[i] * row[idx];
			} else {
				sum += row[idx];
			}
		}
		input[idx] = sigmoid(sum);
	}
}


cuda_mlp::cuda_mlp() {
	weight_list = NULL;
	dev_weight = NULL;
	dev_input = NULL;
	layers_num = 0;
	max_nodes_num = 0;
	kernel_time = 0;
	load_time = 0;
	//cout << "class cuda_mlp is created" << endl;
}

int cuda_mlp::load(const char* file_path){
	//read files;
	ifstream fin(file_path);
	if(fin){
		fin>>layers_num;

		int *nodes_num = new int[layers_num];

		for (int i = 0; i < layers_num + 1; i++) {
			fin >> nodes_num[i];
			if (nodes_num[i] > max_nodes_num)
				max_nodes_num = nodes_num[i];
		}

		weight_list = new Matrix[layers_num];

		for (int i = 0; i < layers_num; i++) {

			weight_list[i].width = nodes_num[i + 1];
			weight_list[i].height = nodes_num[i] + 1;

			int num = weight_list[i].width * weight_list[i].height;
			weight_list[i].elements = new float[num];

			for (int j = 0; j < num; j++) {
				fin >> weight_list[i].elements[j];
			}
		}
	//  cout << "layers_num : " << layers_num << endl;
		delete[] nodes_num;
	}
	else{
		cout<<"can't open mod file"<<endl;
	}
	//malloc for gpu;
	 dev_weight = new Matrix[layers_num];
	for(int i=0;i<layers_num;i++){

		dev_weight[i].width = weight_list[i].width;

		dev_weight[i].height = weight_list[i].height;

		CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&dev_weight[i].elements,
				sizeof(float) * weight_list[i].width*weight_list[i].height));
		CUDA_CHECK_RETURN(
			cudaMemcpy(dev_weight[i].elements, weight_list[i].elements, sizeof(float) * weight_list[i].width * weight_list[i].height,
				cudaMemcpyHostToDevice));

	}
	CUDA_CHECK_RETURN(
		cudaMalloc((void ** )&dev_input,
			sizeof(float) * max_nodes_num));
	return 0;
}

int cuda_mlp::run(const float* input, float* output){
	int output_length = dev_weight[layers_num - 1].width;
	int input_length = dev_weight[0].height - 1;
	CUDA_CHECK_RETURN(
		cudaMemcpy(dev_input, input, sizeof(float) * input_length,
			cudaMemcpyHostToDevice));
	for (int i = 0; i < layers_num ; i++) {
		int blockCount = (dev_weight[i].width + BLOCK_SIZE - 1)
					/ BLOCK_SIZE;
		MatMulKernel<<<blockCount, BLOCK_SIZE,1024>>>(dev_weight[i], dev_input);
	//used for debug;
	}
	CUDA_CHECK_RETURN(
		cudaMemcpy(output, dev_input, sizeof(float) * output_length,
			cudaMemcpyDeviceToHost));
	return 0;
	
}

void cuda_mlp::kernel_free(){
	CUDA_CHECK_RETURN(cudaFree(dev_input));
	for (int i = 0; i < layers_num; i++) {
		delete[] weight_list[i].elements;
		CUDA_CHECK_RETURN(cudaFree(dev_weight[i].elements));
	}
	delete[] weight_list;
	delete[] dev_weight;
}

cuda_mlp::~cuda_mlp(){
	//cout << "class cuda_mlp is deleted" << endl;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
