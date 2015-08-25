#include <iostream>
#include <cstdlib>
#include "index_searcher.h"

int main(int argc, char* argv[])
{
	if (argc != 2 && argc != 4)
	{
		printf("Usage: %s [flags] idx-path\n", argv[0]);

		return 0;
	}

	int argWordCnt = 0;
	int argSentCnt = 0;
	int argParCnt = 0;
	int queryCnt = 1;
	std::string query;
	TIndexSearcher index;
	
	if (argc == 2)
	{
		index.SetDir(argv[1]);
	}
	else
	{
		index.SetDir(argv[3]);

		if (strcmp(argv[1], "-w") == 0)
		{
			argWordCnt = atoi(argv[2]);
		}
		else if (strcmp(argv[1], "-s") == 0)
		{
			argSentCnt = atoi(argv[2]);
		}
		else if (strcmp(argv[1], "-p") == 0)
		{
			argParCnt = atoi(argv[2]);
		}
		else
		{
			printf("ERROR: Unknown parameter\n");

			return 0;
		}
	}
	
	printf("Loading index...\n");

	if (index.Load())
	{
		printf("Index loaded\n");
	}
	else
	{
		printf("ERROR: Can't load index from file\n");
	}

	printf("! idx_searcher started\n");

	while (std::getline(std::cin, query))
	{
		if (query.length() == 0)
		{
			continue;
		}

		printf("! query %d execution: %s\n", queryCnt++, query.c_str());
		
		bool isFound = false;

		if (argWordCnt > 0)
		{
			isFound = index.GetContextWord(query, argWordCnt);
		}
		else if (argSentCnt > 0)
		{
			isFound = index.GetContextSentence(query, argSentCnt);
		}
		else if (argParCnt > 0)
		{
			isFound = index.GetContextParagraph(query, argParCnt);
		}
		else
		{
			TUINT cnt = index.GetResultCount(query);

			if (cnt > 0)
			{
				printf("Matched: %u\n", cnt);
				
				isFound = true;
			}
		}
		
		if (!isFound)
		{
			printf("Not found\n");
		}
	}
	
	return 0;
}
