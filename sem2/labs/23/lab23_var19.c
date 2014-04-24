/*
19. Определить ширину двоичного дерева
*/

#include <stdio.h>
#include "vector.h"
#include "bst.h"

void PKL(BstNode **node, const int level);
void countNodesOnLevels(BstNode **node, Vector *v, const int h);
int max(int a, int b);
int treeDFS(BstNode **node);

int main(void)
{
	int i, maxBFS;
	char cmd[255], arg;
	BstNode *root = NULL;
	Vector v;

	do
	{
		printf("Введите команду (h - справка):\n");
		scanf("%s", cmd);

		if (cmd[0] == '+')
		{
			scanf(" %c", &arg);

			if (arg >= 'A' && arg <= 'Z')
			{
				bstInsert(&root, arg - 'A');

				printf("Узел %c вставлен\n", arg);
			}
			else
				printf("Ошибка. Введена недопустимая буква\n");
		}
		else if (cmd[0] == '-')
		{
			scanf(" %c", &arg);

			if (arg >= 'A' && arg <= 'Z')
			{
				if (bstRemove(&root, arg - 'A'))
					printf("Узел %c удален\n", arg);
				else
					printf("Узел %c не найден\n", arg);
			}
			else
				printf("Ошибка. Введена недопустимая буква\n");
		}
		else if (cmd[0] == 'p')
		{
			PKL(&root, 0);
		}
		else if (cmd[0] == 't')
		{
			if (root != NULL)
			{
				vectorCreate(&v, treeDFS(&root));

				for (i = 0; i < vectorSize(&v); i++)
					vectorSave(&v, i, 0);

				countNodesOnLevels(&root, &v, 0);

				maxBFS = 0;

				for (i = 0; i < vectorSize(&v); i++)
					if (vectorLoad(&v, i) > maxBFS)
						maxBFS = vectorLoad(&v, i);

				printf("Ширина двоичного дерева: %d\n", maxBFS);

				vectorDestroy(&v);
			}
			else
				printf("Двоичное дерево пусто\n");
		}
		else if (cmd[0] == 'h')
		{
			printf("================================\n");
			printf("Список команд:\n");
			printf("+ CHAR - вставить узел CHAR (A, B, ..., Z) в двоичное дерево\n");
			printf("- CHAR - удалить узел CHAR из двоичного дерева\n");
			printf("p - распечатать двоичное дерево\n");
			printf("t - выполнить задание над двоичным деревом\n");
			printf("q - завершить программу\n");
			printf("================================\n");
		}
		else if (cmd[0] != 'q')
		{
			printf("Неизвестная команда\n");
		}
	}
	while (cmd[0] != 'q');

	bstDestroy(&root);

	return 0;
}

void PKL(BstNode **node, const int level)
{
	if (*node == NULL)
	{
		printf("Дерево пусто\n");

		return;
	}

	if ((*node)->_right != NULL)
		PKL(&(*node)->_right, level + 1);

	printf("%*s%c\n", level * 2, "", (*node)->_key + 'A');

	if ((*node)->_left != NULL)
		PKL(&(*node)->_left, level + 1);
}

void countNodesOnLevels(BstNode **node, Vector *v, const int h)
{
	int curH = 0;

	if (*node == NULL)
		return;

	curH = vectorLoad(v, h);

	vectorSave(v, h, curH + 1);

	countNodesOnLevels(&(*node)->_left, v, h + 1);
	countNodesOnLevels(&(*node)->_right, v, h + 1);
}

int max(int a, int b)
{
	return (a > b ? a : b);
}

int treeDFS(BstNode **node)
{
	if (*node == NULL)
		return 0;

	return max(1 + treeDFS(&(*node)->_left), 1 + treeDFS(&(*node)->_right));
}
