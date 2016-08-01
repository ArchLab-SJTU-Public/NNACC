#ifndef APPROXIMATOR_H_INCLUDED
#define APPROXIMATOR_H_INCLUDED
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "NetFactory.h"
using namespace std;

class approximator{
    public:
    static const int MOD_CPU = 0;
    static const int MOD_CUDA = 1;
    static const int MOD_OPENCL = 2;
    static const string MOD_PATH_PREFIX;
    approximator();
    void init(const char* file_path);
    void exec(float* input, float* output, int mod=0);
    void netFree();
    ~approximator();
    private:
    string input_file_path;
    string output_file_path;
    string mod_file_path;
};

#endif // APPROXIMATOR_H_INCLUDED
