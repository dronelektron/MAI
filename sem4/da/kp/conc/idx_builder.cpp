#include <sys/stat.h>
#include "index_builder.h"

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		printf("Usage: %s idx-path file1 file2 ...\n", argv[0]);

		return 0;
	}

	printf("! idx_builder started\n");

	TUINT tokensTotal = 0;
	TUINT termsTotal = 0;
	TIndexBuilder index;

	mkdir(argv[1], 0755);
	
	index.SetDir(argv[1]);
	index.Clear();
	
	for (TUINT i = 2; i < argc; ++i)
	{
		printf("! processing file %s\n", argv[i]);

		TUCHAR docId = i - 2;
		FILE* file = fopen(argv[i], "r");

		if (file == NULL)
		{
			printf("ERROR: Can't open file %s\n", argv[i]);

			continue;
		}

		std::pair<TUINT, TUINT> result = index.Add(file, docId);

		tokensTotal += result.first;
		termsTotal += result.second;

		fclose(file);

		printf("! file %s processed: %u tokens, %u terms\n", argv[i], result.first, result.second);
	}

	printf("Saving index on disk...\n");

	if (index.Save(&argv[2], argc - 2))
	{
		printf("Index saved\n");
	}
	else
	{
		printf("ERROR: Can't save index to file\n");
	}

	printf("! idx_builder finished: total %u tokens, %u terms\n", tokensTotal, termsTotal);

	return 0;
}
