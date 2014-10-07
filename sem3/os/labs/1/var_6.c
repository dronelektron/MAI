#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

typedef enum _kPipe
{
	READ = 0,
	WRITE
} kPipe;

int getStringFromPipe(int* tube);
int checkFile(const char* filename);
void swapStr(char** str1, char** str2);
void sortFile(const char* filename, int* tube);

int main(int argc, char* argv[])
{
	int pipe1[2];
	int pipe2[2];
	pid_t proc1;
	pid_t proc2;

	if (argc < 3)
	{
		printf("Usage: %s <filename 1> <filename 2>\n", argv[0]);

		return 0;
	}

	if (!checkFile(argv[1]) || !checkFile(argv[2]))
		return 0;

	if (pipe(pipe1) == -1)
	{
		printf("ERROR: Can't make pipe 1\n");

		return 0;
	}

	proc1 = fork();

	if (proc1 == -1)
	{
		printf("ERROR: Can't fork child 1\n");

		return 0;
	}

	if (proc1 != 0)
	{
		if (pipe(pipe2) == -1)
		{
			printf("ERROR: Can't make pipe 2\n");

			return 0;
		}

		proc2 = fork();

		if (proc2 == -1)
		{
			printf("ERROR: Can't fork child 2\n");

			return 0;
		}
	}

	if (proc1 == 0)
	{
		close(pipe1[READ]);
		
		sortFile(argv[1], pipe1);

		close(pipe1[WRITE]);
	}
	else if (proc2 == 0)
	{
		close(pipe2[READ]);
		
		sortFile(argv[2], pipe2);

		close(pipe2[WRITE]);
	}
	else
	{
		close(pipe1[WRITE]);
		close(pipe2[WRITE]);

		while (getStringFromPipe(pipe1) && getStringFromPipe(pipe2));
		while (getStringFromPipe(pipe1));
		while (getStringFromPipe(pipe2));

		close(pipe1[READ]);
		close(pipe2[READ]);
	}

	return 0;
}

int getStringFromPipe(int* tube)
{
	char* buffer = NULL;
	unsigned int len = 0;

	if (read(tube[READ], &len, sizeof(len)) == 0)
		return 0;

	buffer = (char*)malloc(sizeof(char) * len);

	read(tube[READ], buffer, len);

	printf("%s\n", buffer);

	free(buffer);

	return 1;
}

int checkFile(const char* filename)
{
	FILE* file = fopen(filename, "r");

	if (file == NULL)
	{
		printf("ERROR: Can't open file '%s'\n", filename);

		return 0;
	}

	fclose(file);

	return 1;
}

void swapStr(char** str1, char** str2)
{
	char* str = *str1;
	
	*str1 = *str2;
	*str2 = str;
}

void sortFile(const char* filename, int* tube)
{
	int ch;
	int i, j;
	unsigned int linesCount = 0;
	unsigned int cnt = 0;
	char** lines = NULL;
	FILE* file = fopen(filename, "r");

	fseek(file, 0, SEEK_END);

	if (ftell(file) > 0)
		linesCount++;
	
	fseek(file, 0, SEEK_SET);

	while ((ch = getc(file)) != EOF)
		if (ch == '\n')
			linesCount++;

	if (ftell(file) > 0)
	{
		fseek(file, ftell(file) - 1, SEEK_SET);

		if (getc(file) == '\n')
			linesCount--;
	}

	if (linesCount == 0)
	{
		fclose(file);

		return;
	}

	lines = (char**)malloc(sizeof(char*) * linesCount);

	fseek(file, 0, SEEK_SET);

	for (i = 0; i < linesCount; ++i)
	{
		cnt = 0;

		while (1)
		{
			ch = getc(file);

			if (ch == '\n' || ch == EOF)
			{
				lines[i] = (char*)malloc(sizeof(char) * (cnt + 1));

				break;
			}

			cnt++;
		}
	}

	fseek(file, 0, SEEK_SET);

	for (i = 0; i < linesCount; ++i)
	{
		j = 0;

		while (1)
		{
			ch = getc(file);

			if (ch == '\n' || ch == EOF)
				break;

			lines[i][j++] = (char)ch;
		}

		lines[i][j] = '\0';
	}
	
	for (i = 0; i < linesCount; ++i)
		for (j = 0; j < linesCount - i - 1; ++j)
			if (strcmp(lines[j], lines[j + 1]) > 0)
				swapStr(&lines[j], &lines[j + 1]);

	for (i = 0; i < linesCount; ++i)
	{
		cnt = strlen(lines[i]) + 1;

		write(tube[WRITE], &cnt, sizeof(cnt));
		write(tube[WRITE], lines[i], cnt);

		free(lines[i]);
	}

	free(lines);
	fclose(file);
}
