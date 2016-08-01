#ifndef NETFACTORY_H
#define NETFACTORY_H

#include <iostream>
#include <map>
#include <string>
#include "net.h"
#include "cpu_mlp.h"
//#include "cuda_mlp.h"
#include "cl_mlp.h"
using namespace std;

class NetFactory{
public:

	static NetFactory& getInstance();

	net* getNet(const char* filepath, const int mod);

	void netFree();

private:

	map<string, net*> cpu_map;
	map<string, net*> cuda_map;
	map<string, net*> cl_map;

	NetFactory();

	~NetFactory();

	NetFactory(NetFactory const&);

	void operator=(NetFactory const&);

};

#endif
