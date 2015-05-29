#include <cstdio>
#include "big_integer.h"

int main()
{
	char arg[100001];
	char ch;
		
	while (scanf("%s", arg) == 1)
	{
		TBigInteger a(arg);

		scanf("%s", arg);
		
		TBigInteger b(arg);
		TBigInteger c;
		
		getchar();
		ch = getchar();
		getchar();
		
		switch (ch)
		{
			case '+':
			{
				c = a + b;
				c.Print();

				break;
			}

			case '-':
			{
				if (a < b)
				{
					printf("Error\n");
				}
				else
				{
					c = a - b;
					c.Print();
				}

				break;
			}

			case '*':
			{
				c = a * b;
				c.Print();

				break;
			}

			case '/':
			{
				if (b == 0)
				{
					printf("Error\n");
				}
				else
				{
					c = a / b;
					c.Print();
				}
				
				break;
			}

			case '^':
			{
				if (a == 0 && b == 0)
				{
					printf("Error\n");
				}
				else if (a == 0)
				{
					printf("0\n");
				}
				else
				{
					c = a ^ b;
					c.Print();
				}
				
				break;
			}

			case '<':
			{
				printf("%s\n", a < b ? "true" : "false");

				break;
			}

			case '>':
			{
				printf("%s\n", a > b ? "true" : "false");

				break;
			}

			case '=':
			{
				printf("%s\n", a == b ? "true" : "false");

				break;
			}
		}
	}
	
	return 0;
}
