OPENCV=1

CXX:= g++ =std=c++17

NVCC:=/usr/local/cuda/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda/include

ifeq ($(OPENCV), 1)
	COMMON= -DOPENCV
	CFLAGS+= $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
	LIBS+= $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

LIBS+= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.hpp)

SRCFILES:= $(filter-out calibrator.cpp, $(wildcard *.cpp))

ifeq ($(OPENCV), 1)
	SRCFILES+= calibrator.cpp
endif

SRCFILES+= $(wildcard layers/*.cpp)
SRCFILES+= $(wildcard *.cu)

TARGET_LIB:= libnvdsinfer_custom_parser_yolo.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
