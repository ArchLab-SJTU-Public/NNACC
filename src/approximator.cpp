#include <approximator.h>

const string approximator::MOD_PATH_PREFIX = "../mod/";

approximator::approximator()
{}

approximator::~approximator()
{}


void approximator::init(const char* file_path)
{
	cout<<"searching mod file:"<<endl;
    ifstream fin;
    fin.open(file_path,ios::in);
    getline(fin,input_file_path);
    getline(fin,output_file_path);
    getline(fin,mod_file_path);
    mod_file_path = MOD_PATH_PREFIX+mod_file_path+".mod";
    fstream min(mod_file_path.c_str());
    
    if(!min){
        cout<<"no mod file found!"<<endl;
		cout<<"start training:"<<endl;
        string str = "python train.py "+input_file_path+" "+output_file_path+" "+mod_file_path;
        system(str.c_str());
		cout<<"training done!"<<endl;        
    }
    else{
        cout<<"mod file found!"<<endl;
    }
}

void approximator::exec(float* input, float* output, int mod)
{
	if(mod == MOD_CPU)
	{
		NetFactory& factory = NetFactory::getInstance();
		net* net_instance = factory.getNet(mod_file_path.c_str(),mod);
		cout<<"exec using cpu:"<<endl;
		net_instance->run(input,output);
	}

	else if(mod == MOD_CUDA)
	{
		NetFactory& factory = NetFactory::getInstance();
		net* net_instance = factory.getNet(mod_file_path.c_str(),mod);
		cout<<"exec using CUDA backend:"<<endl;
		net_instance->run(input,output);
	}

	else if(mod == MOD_OPENCL)
	{
		NetFactory& factory = NetFactory::getInstance();
		net* net_instance = factory.getNet(mod_file_path.c_str(),mod);
		cout<<"exec using OpenCL backend"<<endl;
		net_instance->run(input,output);
	}	
}


void approximator::netFree()
{
	NetFactory& factory = NetFactory::getInstance();
	factory.netFree();
}
