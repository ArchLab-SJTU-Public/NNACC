#NNACC
##Documentation

###1 Project Layout
At the root directory of this project, there are 9 sub-directories, their name and function are describe below:

`bin`:the final output binary<br>`src`:the source files for the framework(exclude the backend parts)<br>`backend`:the module to execute the NN on different platforms<br>`include`: head files<br>`py_Train`:the training scripts used by the framework<br>`doc`: documentations<br>`mod`:the position storing NN config files

Below are the directory you may use when using this Framework:

`conf`: Put the configure file of the application. In this file you should at lease appoint where to find the input/output data of this application<br>`workspce`: this is where you put your source code, to work together with our framework.

###2 Usage
We assume this Project to work somehow like a tool. That means, Users put their code at a certain location, then the framework take over the control and do the execution.

To be exact, put your source code in the `workspace` directory, and you should include `<approximate.h>`. then you initialize the instance with a
configure file, which you should put into the `/conf` directory, for more
about the syntax of this configure file, please refer to the section
config in the `/doc` directory.

The `init` method will either pickup an cached Net Config or invoke a
training script to train and pick out a Net config for you, then the NN
module is ready for execution.

Then everytime you want to use this NN to compute the output to certain
input data, just call the exec method. whose interface is:

```cpp
int approximator::exec(float* input, float* output, int mod)
```
The mod value indicate the execution backend, current the framework
support CPU, cuda and OpenCL. To specify, you can use:   
`MOD_CPU`, `MOD_CUDA`, `MOD_OPENCL`

Finally, when your program is to exit, you need to manually release the
resources by calling `netFree()` method.

###3 Project Skeleton
This Project is basically made up of two parts: The front-end class `Approximator`, whose responsibility is to generate the text file which store a certain NN configure to the application. And the back-end class `NetFactroy` which possess several back-end module to execute the NN on different processors. 

To elaborate, after initialize an instance of approximator, the class either pick an cached NN configuration, or invoke the python script to train several NN (current MLP) and pickup the best basing on the Input/Output file user provided. After the NN configure is Ready, the control is passed to the backend.

The Backend class `NetFactory` actually manage several backend modules(classes), this class realize the so called 'factory mode' to make sure there is only one single instance of a certain class with regard to a certain configure file. This design make sure that there won't be multi instance to perform the exact task, thus eliminate the waste of memory and the overhead of duplicate loads.

All the backend module are abstracted by a virtual class `Net`, which is defined as:
```cpp
class net
{
public:
	virtual int run(const float* input, float* output) = 0;
	virtual int load(const char* file_path) = 0;
	virtual	void kernel_free() = 0; 
};
```
the backend modules all inherit from this virtual class to guarantee an uniformed interface and thus make the backend expandable.

###4 Develop Guide
#### Add Backend 
Adding more backend support is straight forward. First, develop a class targeting at your hardware and realize the interface defined by class `net`. Then change the `/include/Netfoctory.h` to add map to contain your class. And in the `/src/NetFactory.cpp` to copy the code for cuda/cl/cpu to support your class. Finally, define a indicator in `/include/approximator.h` for your backend so that you can call it when using the `approximator::exec` function.

After all this are done, put the cpp file for your class in the `backend` directory and .h file in the `include` directory, then revise the Makefile to add rules to compile your module. Remake the project, your new module will be ready for use.
