OBJS = approximator.o NetFactory.o test.o
BACKEND_OBJS = cl_mlp.o cpu_mlp.o cuda_mlp.o
TARGET = test
OBJDIR = bin
VPATH = src:backend:.:include:workspace:bin
INCLUDEDIR = include


CUDA_DEPS =-L/usr/local/cuda/lib64/ -lcudart
CL_DEPS =-L/usr/local/cuda/lib64/ -lOpenCL
CL_HEAD = -I/usr/local/cuda/include/ 

test:$(OBJS) $(BACKEND_OBJS) $(OBJDIR)
	cd $(OBJDIR) && g++ $(OBJS) $(BACKEND_OBJS) $(CL_DEPS) $(CUDA_DEPS)  -o $@

$(OBJDIR):
	mkdir -p ./$@

$(OBJS):%.o:%.cpp 
	$(CXX) -c -I$(INCLUDEDIR) $(CL_HEAD) $^ -o $@
	mv $@ ./$(OBJDIR)

cl_mlp.o:./backend/cl_mlp.cpp $(OBJDIR)
	$(CXX) -c -I$(INCLUDEDIR) $(CL_HEAD) $^ -o $@
	mv $@ ./$(OBJDIR)

cpu_mlp.o:./backend/cpu_mlp.cpp $(OBJDIR)
	$(CXX) -c -I$(INCLUDEDIR) $^ -o $@
	mv $@ ./$(OBJDIR)

cuda_mlp.o:./backend/cuda_mlp.cu $(OBJDIR)
	nvcc -c  -I$(INCLUDEDIR) $(CL_HEAD) $<  -gencode arch=compute_52,code=sm_52 -o $@
	mv $@ ./$(OBJDIR)
clean:
	$(RM) $(OBJDIR)/*
