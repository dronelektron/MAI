/*
Лабораторная работа 11
Студента: Барковского А.А.
Группа: 80-107Б
*/

#include <stdio.h>
#include <limits.h>

typedef enum _kState
{
	READ = 0,
	SKIP,
	IS_OVERFLOW,
	OVERFLOW,
	MINUS
} kState;

int isLetter(int ch);
int isNumber(int ch);
int isSpace(int ch);

int main(void)
{
	kState state = READ;
	int num = 0, len = 0, isNumGen = 0, ch;
	char buffer[81] = {'\0'};

	while ((ch = getchar()) != EOF)
	{
		switch (state)
		{
			case READ:
			{
				if (isNumber(ch))
				{
					num = num * 10 + ch - '0';
					isNumGen = 1;

					if (num == INT_MAX / 10) state = IS_OVERFLOW;
					else if (num > INT_MAX / 10) state = OVERFLOW;
				}
				else
				{
					if (ch == '-')
					{
						if (isNumGen)
						{
							printf("%d-", num);

							num = 0;
							isNumGen = 0;

							state = SKIP;
						}
						else
						{
							buffer[len++] = '-';
							buffer[len] = '\0';
							state = MINUS;
						}
					}
					else if (!isSpace(ch))
					{
						if (isNumGen) printf("%d", num);

						printf("%c", ch);

						state = SKIP;
					}

					num = 0;
					isNumGen = 0;
				}
			}
			break;

			case SKIP:
			{
				if (isSpace(ch))
				{
					printf(" ");

					state = READ;
				}
				else printf("%c", ch);
			}
			break;

			case IS_OVERFLOW:
			{
				state = READ;

				if (isLetter(ch) || (isNumber(ch) && (ch > INT_MAX % 10 + '0')))
				{
					printf("%d%c", num, ch);

					num = 0;
					isNumGen = 0;
					state = SKIP;
				}
			}
			break;

			case OVERFLOW:
			{
				state = READ;

				if (!isSpace(ch))
				{
					printf("%d%c", num, ch);

					state = SKIP;
				}

				num = 0;
				isNumGen = 0;
			}
			break;

			case MINUS:
			{
				if (!isSpace(ch))
				{
					if (!isNumber(ch))
					{
						if (isNumGen) printf("%s%c", buffer, ch);
						else printf("-%c", ch);

						len = 0;
						buffer[len] = '\0';
						isNumGen = 0;
						state = SKIP;
					}
					else
					{
						buffer[len++] = ch;
						buffer[len] = '\0';
						isNumGen = 1;
					}
				}
				else
				{
					if (!isNumGen) printf("- ");

					len = 0;
					buffer[len] = '\0';
					isNumGen = 0;
					state = READ;
				}
			}
			break;
		}

		if (ch == 10) printf("\n");
	}

	return 0;
}

int isLetter(int ch)
{
	return ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

int isNumber(int ch)
{
	return (ch >= '0' && ch <= '9');
}

int isSpace(int ch)
{
	return (ch == ' ' || ch == ',' || ch == '\n' || ch == '\t');
}
