#include <approximator.h>

int main()
{
    approximator app;
    app.init("sample.config");
    float in[2] = {1,1};
    float out[1];
   // app.exec(in,out,approximator::MOD_GPU);
    app.exec(in,out,approximator::MOD_OPENCL);
    cout<<out[0]<<endl;
    app.netFree();
    return 0;
}
