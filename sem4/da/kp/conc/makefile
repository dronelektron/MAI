CC = g++
CFLAGS = -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -Wno-unused-result -lm -O2
SHARED_FILES = tokenizer.cpp index.cpp

FILES_1 = idx_builder.cpp index_builder.cpp $(SHARED_FILES)
PROG_1 = idx_builder

FILES_2 = idx_searcher.cpp index_searcher.cpp $(SHARED_FILES)
PROG_2 = idx_searcher

all:
	$(CC) $(CFLAGS) -o $(PROG_1) $(FILES_1)
	$(CC) $(CFLAGS) -o $(PROG_2) $(FILES_2)

clean:
	rm $(PROG_1) $(PROG_2)
