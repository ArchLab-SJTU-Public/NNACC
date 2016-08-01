#include "../include/cpu_mlp.h"

float cpu_mlp::non_linear(float x)
{
	return 1.0 / (1.0 + exp(0.0 - x));
}

cpu_mlp::cpu_mlp()
{
	max_dim = 0;
	total_item = 0; 
	layer_num = 0;
	h_layer_dim = NULL;
	h_net = NULL;
}

int cpu_mlp::load(const char* net_file)
{
	FILE *net_f;	
	int next_dim, curr_dim;
	int offset = 0;
	int tmp,i;

	net_f = fopen(net_file, "r");
	fscanf(net_f, "%d", &layer_num);
	h_layer_dim = (int*)malloc(sizeof(int)*layer_num);
	fscanf(net_f, "%d", &curr_dim);
	max_dim = curr_dim;
	h_layer_dim[0] = curr_dim;
	for (i = 0; i < (layer_num - 1); i++)
	{
		fscanf(net_f, "%d", &next_dim);
		if (next_dim > max_dim)
		{
			max_dim = next_dim;
		}
		h_layer_dim[i + 1] = next_dim;
		total_item += (curr_dim + 1)*next_dim;
		curr_dim = next_dim;
	}
	h_net = (float*)malloc(sizeof(float)*total_item);
	for (i = 0; i<total_item; i++)
	{
		fscanf(net_f, "%f", &h_net[i]);
	}
	fclose(net_f);

	h_inter_res[0] = (float*)malloc(sizeof(float)*(max_dim + 1));
	h_inter_res[1] = (float*)malloc(sizeof(float)*(max_dim + 1));
}

int cpu_mlp::run(const float* input, float* output)
{
	//#ifdef WIN
	//LARGE_INTEGER large_int;
	//double diff;
	//__int64 c1, c2;

	//QueryPerformanceFrequency(&large_int);
	//diff = large_int.QuadPart;
	//QueryPerformanceCounter(&large_int);
	//c1 = large_int.QuadPart;
	//#endif
	//#ifdef UNIX
	//long c1, c2;
	//struct timeval tv;
	//gettimeofday(&tv, NULL);
	//c1 = tv.tv_usec;
	//#endif	

	float tmp_res;
	int offset,tmp;
	int i,j,k;
	int curr_dim, next_dim;
	int data_pool, result_pool;
	
	for(i=0;i<h_layer_dim[0];i++)
	{
		h_inter_res[0][i] = input[i];
	}
	h_inter_res[0][h_layer_dim[0]] = 1.0;

	offset = 0;
	for(i=0;i<layer_num-1;i++)
	{
		curr_dim = h_layer_dim[i] + 1;
		next_dim = h_layer_dim[i+1];
		
		//this is a very smart design, avoid branch to decide which buffer to use
		data_pool = i%2;
		result_pool = (i+1)%2;
		for(j = 0;j<next_dim;j++)
		{
			tmp_res = 0.0;
			for(k = 0;k<curr_dim;k++)
			{
				tmp_res += h_inter_res[data_pool][k]*h_net[offset + next_dim*k + j];
			}
			h_inter_res[result_pool][j] = non_linear(tmp_res);				
		}
		h_inter_res[result_pool][next_dim] = 1.0;
		offset += curr_dim*next_dim;
	}

	if(layer_num%2 == 0)
	{
		for (i = 0; i < h_layer_dim[layer_num - 1]; i++)
		{
			output[i] = h_inter_res[1][i];
		}
	}
	else
	{
		for (i = 0; i < h_layer_dim[layer_num - 1]; i++)
		{
			output[i] = h_inter_res[0][i];
		}
	}

	//#ifdef WIN
	//QueryPerformanceCounter(&large_int);
	//c2 = large_int.QuadPart;
	//CPUtime = (float)((c2 - c1)*1e06 / diff);
	//#endif
	//#ifdef UNIX
	//gettimeofday(&tv, NULL);
	//c2 = tv.tv_usec;
	//CPUtime = float(c2 - c1);
	//#endif
	return 0;
}

cpu_mlp::~cpu_mlp()
{
	free(h_net);
	free(h_layer_dim);
	free(h_inter_res[0]);
	free(h_inter_res[1]);	
}