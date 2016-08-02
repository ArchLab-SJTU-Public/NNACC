#ifndef NET_H
#define NET_H

class net
{
public:
	virtual int run(const float* input, float* output) = 0;
	virtual int load(const char* file_path) = 0;
	virtual	void kernel_free() = 0; 
};

#endif
