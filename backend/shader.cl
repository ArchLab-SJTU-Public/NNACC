__kernel void vmm(const int curr_dim,const int next_dim, __global const float* net, __global float* input, __global float* output)
{
	
	int i, j;
	float res = 0.0;
	i = get_global_id(0);
	for(j = 0;j<curr_dim+1;j++)
	{
		res += input[j]*net[j*next_dim + i];
	}
	output[i] = 1.0/(1+ exp(-res));
	if (i == 0)
	{
		output[next_dim] = 1.0;
	}
}