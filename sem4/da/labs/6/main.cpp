#include <cstdio>
#include "big_integer.h"

int main()
{
	char arg[100001];
		
	while (scanf("%s", arg) == 1)
	{
		TBigInteger a(arg);

		if (scanf("%s", arg) != 1)
		{
			printf("ERROR\n");

			return 0;
		}
		
		TBigInteger b(arg);
		TBigInteger c;
		
		if (scanf("%s", arg) != 1)
		{
			printf("ERROR\n");

			return 0;
		}
		
		switch (arg[0])
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
