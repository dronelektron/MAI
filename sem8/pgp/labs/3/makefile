CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
FILES = main.cu image.cu matrix.cu classifier.cu
PROG = lab3

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
