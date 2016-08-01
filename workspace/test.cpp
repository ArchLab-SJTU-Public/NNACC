#include <approximator.h>

int main()
{
    approximator app;
    app.init("../conf/sample.config");
    float in[2] = {1,1};
    float out[1];
   // app.exec(in,out,approximator::MOD_GPU);
    app.exec(in,out,approximator::MOD_CPU);
    cout<<out[0]<<endl;
    app.netFree();
    return 0;
}
