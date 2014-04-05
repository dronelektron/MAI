/*
34. Определить уровень дерева, на котором
находится максимальное число вершин.
*/

#include <stdio.h>
#include "vector.h"
#include "tree.h"

int treeNodesCount(TreeNode **node);
void KLP(TreeNode **node, const int level);
void countNodesOnLevels(TreeNode **node, Vector *levels, const int level);
TreeNode *getNodeByPath(TreeNode **node, const char *path);

int main(void)
{
	int i, maxBFS, levelOfMaxBFS;
	char cmd[81], arg;
	TreeNode *root = NULL, *tmpNode = NULL;
	Vector v;

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
					treeAddNode(&root, arg - 'A');

					printf("Корень %c создан\n", arg);
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
				else if (treeAddNode(&tmpNode, arg - 'A') != NULL)
					printf("Узел %c добавлен к узлу %c\n", arg, tmpNode->_data + 'A');
			}
		}
		else if (cmd[0] == '-')
		{
			scanf(" %c", &arg);

			if (treeRemoveNode(&root, arg - 'A'))
				printf("Узел %c удален\n", arg);
			else
				printf("Узел %c не найден\n", arg);
		}
		else if (cmd[0] == 'p')
		{
			KLP(&root, 0);
		}
		else if (cmd[0] == 't')
		{
			if (root != NULL)
			{
				vectorCreate(&v, treeNodesCount(&root));

				for (i = 0; i < vectorSize(&v); i++)
					vectorSave(&v, i, 0);

				countNodesOnLevels(&root, &v, 0);

				maxBFS = 1;
				levelOfMaxBFS = 1;

				for (i = 1; i < vectorSize(&v); i++)
					if (vectorLoad(&v, i) > maxBFS)
					{
						maxBFS = vectorLoad(&v, i);
						levelOfMaxBFS = i + 1;
					}

				printf("Уровень дерева, на котором число вершин максимально: %d\n", levelOfMaxBFS);

				vectorDestroy(&v);
			}
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

	// DEBUG
	/*
	printf("%*s%d (parent: %d, olderBro: %d, son: %d, bro: %d)\n", level * 2, "",
		(*node)->_data,
		(*node)->_parent != NULL ? (*node)->_parent->_data : -1,
		(*node)->_olderBro != NULL ? (*node)->_olderBro->_data : -1,
		(*node)->_son != NULL ? (*node)->_son->_data : -1,
		(*node)->_bro != NULL ? (*node)->_bro->_data : -1
	);
	*/

	printf("%*s%c\n", level * 2, "", (*node)->_data + 'A');

	if ((*node)->_son != NULL)
		KLP(&(*node)->_son, level + 1);

	if ((*node)->_bro != NULL)
		KLP(&(*node)->_bro, level);
}

int treeNodesCount(TreeNode **node)
{
	int cnt = 0;

	if (*node == NULL)
		return 0;

	if ((*node)->_bro != NULL)
		cnt += treeNodesCount(&(*node)->_bro);

	if ((*node)->_son != NULL)
		cnt += treeNodesCount(&(*node)->_son);

	return cnt + 1;
}

void countNodesOnLevels(TreeNode **node, Vector *levels, const int level)
{
	if (*node == NULL)
		return;

	vectorSave(levels, level, vectorLoad(levels, level) + 1);

	if ((*node)->_bro)
		countNodesOnLevels(&(*node)->_bro, levels, level);

	if ((*node)->_son)
		countNodesOnLevels(&(*node)->_son, levels, level + 1);
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
