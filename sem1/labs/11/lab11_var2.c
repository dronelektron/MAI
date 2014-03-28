#include <stdio.h>

typedef enum _kState
{
	FIND_1 = 0,
	FIND_2,
	COUNTING	
} kState;

int isSpace(int ch);

int main(void)
{
	int ch, isInput = 0, cntInput = 0, cnt = 0, cntStar = 0;
	kState state = FIND_1;

	while ((ch = getchar()) != EOF)
	{
		switch (state)
		{
			case FIND_1:
			{
				if (ch == '/') state = FIND_2;
			}
			break;

			case FIND_2:
			{
				if (ch == '*') state = COUNTING;
				else if (ch != '/') state = FIND_1;
			}
			break;

			case COUNTING:
			{
				if (!isSpace(ch))
				{
					if (!isInput)
					{
						isInput = 1;
						cnt++;
					}

					if (ch == '*') cntStar++;
					else
					{
						if (ch == '/')
						{
							if ((!isInput || !cntInput) && cntStar < 2) cnt--;

							state = FIND_1;
						}

						cntInput = 1;
						cntStar = 0;
					}
				}
				else
				{
					isInput = 0;
					cntStar = 0;
				}
			}
			break;
		}
	}

	printf("Количество слов: %d\n", cnt);

	return 0;
}

int isSpace(int ch)
{
	return (ch == ' ' || ch == ',' || ch == '\n' || ch == '\t');
}
