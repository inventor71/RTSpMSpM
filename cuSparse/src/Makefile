CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse -lnvToolsExt
LIBDIRS      := -L$(CUDA_TOOLKIT)/targets/x86_64-linux/lib

# Uncomment the following line to enable debug mode
# CFLAGS += -DREUSE

all: cuSparse

cuSparse: main.cpp
	nvcc -g -G $(INC) $(CFLAGS) $(LIBDIRS) main.cpp -o cuSparse $(LIBS)

clean:
	rm -f cuSparse

test:
	@echo "\n==== SpSM COO Test ====\n"
	./cuSparse

.PHONY: clean all test