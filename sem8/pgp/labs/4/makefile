CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
FILES = main.cu gauss.cu
PROG = lab4

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
