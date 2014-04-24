/*
20. Определить глубину максимальной вершины дерева
*/

#include <stdio.h>
#include "tree.h"

void KLP(TreeNode **node, const int level);
int min(int a, int b);
int max(int a, int b);
int treeMaxNode(TreeNode **node);
int treeNodeDFS(TreeNode **node, const int value, const int level);
TreeNode *getNodeByPath(TreeNode **node, const char *path);

int main(void)
{
	int i;
	char cmd[255], arg;
	TreeNode *root = NULL, *tmpNode = NULL;

	do
	{
		printf("Введите команду (h - справка):\n");
		scanf("%s", cmd);

		if (cmd[0] == '+')
		{
			scanf(" %c", &arg);

			if (cmd[1] == 'r')
			{
				if (root == NULL)
				{
					if (arg >= 'A' && arg <= 'Z')
					{
						treeAddNode(&root, arg - 'A');

						printf("Корень %c создан\n", arg);
					}
					else
						printf("Ошибка. Введена недопустимая буква\n");
				}
				else
					printf("Корень уже существует\n");
			}
			else if (root == NULL)
				printf("Корень не создан\n");
			else
			{
				tmpNode = root;

				if (cmd[1] != '\0')
					tmpNode = getNodeByPath(&root, &cmd[1]);

				if (tmpNode == NULL)
					printf("Ошибка. Такого пути не существует\n");
				else if (arg >= 'A' && arg <= 'Z')
				{
					if (treeAddNode(&tmpNode, arg - 'A') != NULL)
						printf("Узел %c добавлен к узлу %c\n", arg, tmpNode->_data + 'A');
				}
				else
					printf("Ошибка. Введена недопустимая буква\n");
			}
		}
		else if (cmd[0] == '-')
		{
			scanf(" %c", &arg);

			if (arg >= 'A' && arg <= 'Z')
			{
				if (treeRemoveNode(&root, arg - 'A'))
					printf("Узел %c удален\n", arg);
				else
					printf("Узел %c не найден\n", arg);
			}
			else
				printf("Ошибка. Введена недопустимая буква\n");
		}
		else if (cmd[0] == 'p')
		{
			KLP(&root, 0);
		}
		else if (cmd[0] == 't')
		{
			if (root != NULL)
				printf("Глубина максимальной вершины дерева равна: %d\n", treeNodeDFS(&root, treeMaxNode(&root), 1));
			else
				printf("Дерево пусто\n");
		}
		else if (cmd[0] == 'h')
		{
			printf("================================\n");
			printf("Список команд:\n");
			printf("+r CHAR - создать корень CHAR (A, B, ..., Z)\n");
			printf("+ CHAR - добавить сына CHAR к корню\n");
			printf("+PATH CHAR - добавить CHAR узел по заданому пути (s - сын, b - брат)\n");
			printf("- CHAR - удалить первый найденный узел CHAR и его поддерево\n");
			printf("p - распечатать дерево\n");
			printf("t - выполнить задание над деревом\n");
			printf("q - завершить программу\n");
			printf("================================\n");
		}
		else if (cmd[0] != 'q')
		{
			printf("Неизвестная команда\n");
		}
	}
	while (cmd[0] != 'q');

	treeDestroy(&root);

	return 0;
}

void KLP(TreeNode **node, const int level)
{
	if (*node == NULL)
	{
		printf("Дерево пусто\n");

		return;
	}

	printf("%*s%c\n", level * 2, "", (*node)->_data + 'A');

	if ((*node)->_son != NULL)
		KLP(&(*node)->_son, level + 1);

	if ((*node)->_bro != NULL)
		KLP(&(*node)->_bro, level);
}

int min(int a, int b)
{
	return (a < b ? a : b);
}

int max(int a, int b)
{
	return (a > b ? a : b);
}

int treeMaxNode(TreeNode **node)
{
	if (*node == NULL)
		return 0;

	if ((*node)->_bro == NULL && (*node)->_son == NULL)
		return (*node)->_data;

	return max(treeMaxNode(&(*node)->_son), treeMaxNode(&(*node)->_bro));
}

int treeNodeDFS(TreeNode **node, const int value, const int level)
{
	int maxLevel = 0;

	if (*node == NULL)
		return 0;

	if ((*node)->_data == value)
		return level;

	maxLevel = treeNodeDFS(&(*node)->_bro, value, level);

	if (maxLevel)
		return maxLevel;

	return treeNodeDFS(&(*node)->_son, value, level + 1);
}

TreeNode *getNodeByPath(TreeNode **node, const char *path)
{
	int i = 0;
	TreeNode *tmpNode = *node;

	while (path[i] != '\0')
	{
		if (path[i] == 's')
			if (tmpNode->_son != NULL)
				tmpNode = tmpNode->_son;
			else
				return NULL;
		else if (path[i] == 'b')
			if (tmpNode->_bro != NULL)
				tmpNode = tmpNode->_bro;
			else
				return NULL;
		else
			return NULL;

		i++;
	}

	return tmpNode;
}
