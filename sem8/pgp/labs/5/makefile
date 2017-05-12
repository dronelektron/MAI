CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
FILES = main.cu array.cu
PROG = lab5

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
