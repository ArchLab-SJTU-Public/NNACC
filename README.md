# NNACC
###Acceleration Framework Using NN Method

=======
This tool realize a functional interface for calling NN algorithms: Just
provide a set of input-output data pairs as the behavior description of a
module. Then the framework will automatically pick up an Neural Network
configuration replace the original module. The NN can be executed on
multi-backend for the purpose of acceleration.<\br>


###A Brief Instruction to Use this Framework:

After Download and unpack, put your code into the` /workplace` directory.
In your code, you should include `approximator.h` so that you can use the
API we provided. To use the functionality, you first need to make an
instance of class approximator, then you initialize the instance with a
configure file, which you should put into the `/conf` directory, for more
about the syntax of this configure file, please refer to the section
config in the `/doc` directory.<\br>
The init method will either pickup an cached Net Config or invoke a
training script to train and pick out a Net config for you, then the NN
module is ready for execution.<\br>
Then everytime you want to use this NN to compute the output to certain
input data, just call the exec method. whose interface is:<\br>
```cpp
int approximator::exec(float* input, float* output, int mod)
```
the mod value indicate the execution backend, current the framework
support CPU, cuda and OpenCL. To specify, you can use:<\br>
`MOD_CPU`<\br>
`MOD_CUDA`<\br>
`MOD_OPENCL`<\br>
if you use backends like GPUs, the NN config actually need to be moved
to the device memory to be executed. With our design, this data
transfer will only happend once, thus eliminate the overhead of data
transfer<\br>
Finally, when your program is to exit, you need to manually release the
resources by calling `the netFree()` method.

For more detailed description and develop helpers, please refer to the
`/doc` directory.


