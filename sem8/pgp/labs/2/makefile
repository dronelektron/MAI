CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
FILES = main.cu image.cu filter.cu
PROG = lab2

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
