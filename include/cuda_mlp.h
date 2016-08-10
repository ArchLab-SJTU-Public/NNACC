#ifndef CUDA_MLP_H
#define CUDA_MLP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "net.h"
typedef struct {
    int width;
    int height;
    size_t pitch;
    float* elements;
} Matrix;


class cuda_mlp: public net
{
public:
    cuda_mlp();
    ~cuda_mlp();
    static const int BLOCK_SIZE = 256;

    int load(const char* net_file);
    int run(const float* input, float* output);

    void kernel_free();


private:
    Matrix* weight_list;
    Matrix* dev_weight;
    float* dev_input;
    int layers_num;
    int max_nodes_num;
    float kernel_time;
    float load_time;

};

#endif
