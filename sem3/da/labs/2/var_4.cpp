#include <cstdio>
#include "btree.h"

char LowerCase(char ch);
bool IsLetter(char ch);
void ReadLine(char* cmd, char* textArg, TULL* numArg);

int main()
{
	TBTree btree(5);
	
	while (true)
	{
		char cmd;
		char buffer[257];
		TULL num;

		ReadLine(&cmd, buffer, &num);

		if (cmd == 'Q')
		{
			break;
		}

		switch (cmd)
		{
			case '+':
			{
				TBTree::TData data;

				data.key = buffer;
				data.val = num;
				
				if (btree.Find(buffer) == NULL)
				{
					btree.Insert(data);

					printf("OK\n");
				}
				else
				{
					printf("Exist\n");
				}
				
				break;
			}

			case '-':
			{
				if (btree.Find(buffer) == NULL)
				{
					printf("NoSuchWord\n");
				}
				else
				{
					btree.Erase(buffer);

					printf("OK\n");
				}

				break;
			}

			case 'S':
			{
				if (btree.Serialize(buffer))
				{
					printf("OK\n");
				}
				else
				{
					printf("ERROR: Serialize failed\n");
				}

				break;
			}

			case 'L':
			{
				if (btree.Deserialize(buffer))
				{
					printf("OK\n");
				}
				else
				{
					printf("ERROR: Deserialize failed\n");
				}

				break;
			}

			case 'F':
			{
				TBTree::TData* data = btree.Find(buffer);

				if (data == NULL)
				{
					printf("NoSuchWord\n");
				}
				else
				{
					printf("OK: %llu\n", data->val);
				}

				break;
			}
		}
	}

	return 0;
}

char LowerCase(char ch)
{
	return ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch;
}

bool IsLetter(char ch)
{
	return ch >= 'a' && ch <= 'z';
}

void ReadLine(char* cmd, char* textArg, TULL* numArg)
{
	char ch;
	size_t i = 0;
	
	ch = getchar();

	if (ch == EOF)
	{
		*cmd = 'Q';

		return;
	}

	if (ch == '+' || ch == '-')
	{
		getchar();

		*cmd = ch;

		while (true)
		{
			ch = LowerCase(getchar());

			if (!IsLetter(ch))
			{
				break;
			}

			textArg[i++] = ch;
		}

		textArg[i] = '\0';
		
		if (*cmd == '+')
		{
			*numArg = 0;

			while ((ch = getchar()) != '\n')
			{
				*numArg = (*numArg) * 10 + ch - '0';
			}
		}
	}
	else if (ch == '!')
	{
		getchar();

		textArg[0] = ch;

		while ((ch = getchar()) != ' ')
		{
			textArg[i++] = ch;
		}

		textArg[i] = '\0';
		i = 0;

		*cmd = textArg[0];

		while ((ch = getchar()) != '\n')
		{
			textArg[i++] = ch;
		}

		textArg[i] = '\0';
	}
	else
	{
		*cmd = 'F';
		textArg[0] = LowerCase(ch);
		++i;

		while ((ch = getchar()) != '\n')
		{
			textArg[i++] = LowerCase(ch);
		}

		textArg[i] = '\0';
	}
}
