#include <stdio.h>

int isSpace(int c);

int main(void)
{
	int ch, len = 0, cnt = 0;

	while ((ch = getchar()) != EOF)
	{
		if (isSpace(ch))
		{
			if (len >= 3) cnt++;
			
			len = 0;
		}
		else len++;
	}

	printf("Количество слов, длина которых больше трех равно: %d\n", cnt);

	return 0;
}

int isSpace(int c)
{
	return (c == ' ' || c == ',' || c == '\n' || c == '\t');
}
