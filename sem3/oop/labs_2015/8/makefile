CC = g++
CFLAGS = -std=c++11 -Wall -Werror -Wno-sign-compare -Wno-unused-result -pthread
FILES = main.cpp square.cpp rectangle.cpp trapezoid.cpp figure.cpp allocator.cpp
PROG = lab8

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)

clean:
	rm $(PROG)
