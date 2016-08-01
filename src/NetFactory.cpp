#include <NetFactory.h>
using namespace std;

NetFactory& NetFactory::getInstance(){
	static NetFactory instance;
	return instance;
}

net* NetFactory::getNet(const char* filepath, const int mod)
{
	string key(filepath);
	net* instance = NULL;

	if(mod == 0)
	//in cpu search
	{
		if(cpu_map.find(key) != cpu_map.end())
		{
			instance = cpu_map[key];
		}
		else
		{
			if(key.find(".mod") != string::npos)
			{
				//at present the fw only support mlp, so there 
				//is no problem to direct activate a mlp
				//but when later more instance conf is added, 
				//there should be some mechanism to determine 
				//the NN type from the .mod file
				instance = new cpu_mlp();
				instance->load(filepath);
				cpu_map[key] = instance;
			}
			else{
				cout<<"Error: unknown mod type"<<endl;
			}
		}
		return instance;
	}

	else if(mod == 1)
	//means cuda
	{/*
		if(cuda_map.find(key) != cuda_map.end())
		{
			instance = cuda_map[key];
		}
		else
		{
			if(key.find(".mod") != string::npos)
			{
				instance = new cuda_mlp();
				instance->load(filepath);
				cuda_map[key] = instance;
			}
			else{
				cout<<"Error: unknown mod type"<<endl;
			}
		}
		return instance;
*/
	}

	else if(mod == 2)
	//means opencl
	{
		if(cl_map.find(key) != cl_map.end())
		{
			instance = cl_map[key];
		}
		else
		{
			if(key.find(".mod") != string::npos)
			{
				instance = new cl_mlp();
				instance->load(filepath);
				cl_map[key] = instance;
			}
			else{
				cout<<"Error: unknown mod type"<<endl;
			}
		}
		return instance;
	}

	else
	{
		cout<<"Error resolve mode"<<endl;
	}

}

/*
void NetFactory::netFree(){
	for (map<string, mlp*>::iterator it = net_map.begin(); it != net_map.end(); it++) {
		mlp* net = it->second;
		net->kernel_free();
		delete net;
	}
	net_map.clear();
}
*/

NetFactory::NetFactory(){
	//cout<<"net factory is created"<<endl;
}

NetFactory::~NetFactory(){
	//cout<<"net factory is deleted"<<endl;
}


