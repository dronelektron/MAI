#include <stdio.h>
#include <string.h>
#include "stack.h"

typedef struct _Node
{
	char _varOp;
	int _num;
	struct _Node *_left;
	struct _Node *_right;
} Node;

Node *treeNodeCreate(void);
Node *treeCopy(Node **node);
Node *treeMakeMinus(Node **node);
Node *treeMakeNotMinus(Node **node);
int treeIsMinusNode(Node **node);
void treeBuild(Node **node, Stack *st);
void treeDestroy(Node **node);
void treeMoveMinus(Node **node);
void treeCountMinus(Node **node, int *cnt);
void PKL(Node **node, const int level);
void LKP(Node **node);

int isLetter(const char ch);
int isNumber(const char ch);
int isOp(const char ch);
int isOpHigh(const char op1, const char op2);
void postOrder(const char *str, Stack *st);

int main(void)
{
	int action;
	char expr[255];
	Node *root = NULL, *root2 = NULL;
	Stack stPost;

	while (1)
	{
		printf("Меню:\n");
		printf("1) Ввести выражение\n");
		printf("2) Печать исходного выражения\n");
		printf("3) Печать преобразованного выражения\n");
		printf("4) Печать исходного дерева\n");
		printf("5) Печать преобразованного дерева\n");
		printf("6) Выход\n");
		printf("Выберите действие: ");
		scanf("%d", &action);

		switch (action)
		{
			case 1:
			{
				printf("Введите выражение: ");
				scanf("%s", expr);

				treeDestroy(&root);
				treeDestroy(&root2);
				stackCreate(&stPost);
				postOrder(expr, &stPost);
				treeBuild(&root, &stPost);
				stackDestroy(&stPost);
				
				root2 = treeCopy(&root);
				
				treeMoveMinus(&root2);

				break;
			}

			case 2:
			{
				printf("Исходное выражение: %s\n", expr);

				break;
			}

			case 3:
			{
				LKP(&root2);
				printf("\n");

				break;
			}

			case 4:
			{
				if (root != NULL)
				{
					printf("Дерево исходного выражения\n");
					PKL(&root, 0);
				}
				else
					printf("Дерево исходного выражения пусто\n");

				break;
			}

			case 5:
			{
				if (root2 != NULL)
				{
					printf("Дерево преобразованного выражения\n");
					PKL(&root2, 0);
				}
				else
					printf("Дерево преобразованного выражения пусто\n");

				break;
			}

			case 6: break;

			default:
			{
				printf("Ошибка. Такого пункта меню не существует\n");

				break;
			}
		}

		if (action == 6)
			break;
	}

	treeDestroy(&root);
	treeDestroy(&root2);

	return 0;	
}

Node *treeNodeCreate(void)
{
	Node *tmpNode = (Node *)malloc(sizeof(Node));

	tmpNode->_varOp = '\0';
	tmpNode->_num = 0;
	tmpNode->_left = NULL;
	tmpNode->_right = NULL;

	return tmpNode;
}

Node *treeCopy(Node **node)
{
	Node *tmpNode = NULL;

	if (*node == NULL)
		return NULL;

	tmpNode = treeNodeCreate();
	tmpNode->_varOp = (*node)->_varOp;
	tmpNode->_num = (*node)->_num;
	tmpNode->_left = treeCopy(&(*node)->_left);
	tmpNode->_right = treeCopy(&(*node)->_right);

	return tmpNode;
}

Node *treeMakeMinus(Node **node)
{
	Node *tmpNode = NULL;

	tmpNode = treeNodeCreate();
	tmpNode->_left = treeNodeCreate();
	tmpNode->_right = treeCopy(node);
	tmpNode->_varOp = '-';

	free(*node);

	*node = NULL;

	return tmpNode;
}

Node *treeMakeNotMinus(Node **node)
{
	Node *tmpNode = treeCopy(&(*node)->_right);

	treeDestroy(node);

	return tmpNode;
}

int treeIsMinusNode(Node **node)
{
	if (*node == NULL)
		return 0;

	if ((*node)->_left == NULL || (*node)->_right == NULL)
		return 0;

	return ((*node)->_varOp == '-' && (*node)->_left->_varOp == '\0' && (*node)->_left->_num == 0);
}

void treeBuild(Node **node, Stack *st)
{
	Token token;

	if (stackEmpty(st))
		return;

	token = stackTop(st);

	stackPop(st);

	(*node) = treeNodeCreate();
	(*node)->_varOp = token._varOp;
	(*node)->_num = token._num;

	if (isOp((*node)->_varOp))
	{
		treeBuild(&(*node)->_right, st);
		treeBuild(&(*node)->_left, st);
	}
}

void treeDestroy(Node **node)
{
	if (*node == NULL)
		return;

	if ((*node)->_left != NULL)
		treeDestroy(&(*node)->_left);

	if ((*node)->_right != NULL)
		treeDestroy(&(*node)->_right);

	free(*node);

	*node = NULL;
}

void treeMoveMinus(Node **node)
{
	int minusCnt = 0;

	if (*node == NULL)
		return;

	if ((*node)->_varOp == '*' || (*node)->_varOp == '/')
	{
		treeCountMinus(node, &minusCnt);

		if (minusCnt & 1)
			*node = treeMakeMinus(node);
	}
	else if (isOp((*node)->_varOp))
	{
		treeMoveMinus(&(*node)->_left);
		treeMoveMinus(&(*node)->_right);
	}
}

void treeCountMinus(Node **node, int *cnt)
{
	if (*node == NULL)
		return;

	if (treeIsMinusNode(node))
	{
		*node = treeMakeNotMinus(node);
		(*cnt)++;
	}
	else if ((*node)->_varOp == '*' || (*node)->_varOp == '/')
	{
		treeCountMinus(&(*node)->_left, cnt);
		treeCountMinus(&(*node)->_right, cnt);
	}
	else
	{
		treeMoveMinus(&(*node)->_left);
		treeMoveMinus(&(*node)->_right);
	}
}

void PKL(Node **node, const int level)
{
	if (*node == NULL)
		return;

	if ((*node)->_right != NULL)
		PKL(&(*node)->_right, level + 1);

	if ((*node)->_varOp != '\0')
		printf("%*s%c\n", level * 4, "", (*node)->_varOp);
	else
		printf("%*s%d\n", level * 4, "", (*node)->_num);

	if ((*node)->_left != NULL)
		PKL(&(*node)->_left, level + 1);
}

void LKP(Node **node)
{
	if (*node == NULL)
		return;

	if ((*node)->_left != NULL && !treeIsMinusNode(node))
	{
		if ((*node)->_left->_left != NULL)
			printf("(");
		
		LKP(&(*node)->_left);
		
		if ((*node)->_left->_left != NULL)
			printf(")");
	}

	if ((*node)->_varOp != '\0')
		printf("%c", (*node)->_varOp);
	else
		printf("%d", (*node)->_num);

	if ((*node)->_right != NULL)
	{
		if ((*node)->_right->_left != NULL)
			printf("(");

		LKP(&(*node)->_right);
		
		if ((*node)->_right->_left != NULL)
			printf(")");
	}
}

int isLetter(const char ch)
{
	return ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

int isNumber(const char ch)
{
	return (ch >= '0' && ch <= '9');
}

int isOp(const char ch)
{
	return (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == '(' || ch == ')');
}

int opPrior(const char op)
{
	if (op == '^')
		return 4;

	if (op == '*' || op == '/')
		return 3;

	if (op == '+' || op == '-')
		return 2;

	return 1;
}

int isOpHigh(const char op1, const char op2)
{
	if (op1 == '(' || op2 == '(' || op2 == ')')
		return 0;

	if (op1 == op2 && op2 == '^')
		return 0;
	
	return (opPrior(op1) >= opPrior(op2));
}

void postOrder(const char *str, Stack *st)
{
	int i = 0, isBracket = 0;
	char tmpCh;
	Token tk;
	Stack stOp;

	stackCreate(&stOp);

	tk._varOp = '\0';
	tk._num = 0;

	while (str[i] != '\0')
	{
		if (isLetter(str[i]))
		{
			tk._varOp = str[i];

			stackPush(st, tk);
		}
		else if (isNumber(str[i]))
		{
			tk._varOp = '\0';
			tk._num = tk._num * 10 + str[i] - '0';

			if (str[i + 1] != '.' && !isNumber(str[i + 1]))
			{
				stackPush(st, tk);

				tk._num = 0;
			}
		}
		else if (isOp(str[i]))
		{
			tk._varOp = str[i];

			if (str[i] == ')')
				isBracket = 1;
			else if (str[i] == '-' && (i == 0 || str[i - 1] == '('))
			{
				tmpCh = tk._varOp;
				tk._varOp = '\0';
				tk._num = 0;

				stackPush(st, tk);

				tk._varOp = tmpCh;
			}

			while (!stackEmpty(&stOp) && (isOpHigh(stackTop(&stOp)._varOp, str[i]) || isBracket))
			{
				if (stackTop(&stOp)._varOp == '(')
					isBracket = 0;
				else
					stackPush(st, stackTop(&stOp));

				stackPop(&stOp);
			}

			if (str[i] != ')')
				stackPush(&stOp, tk);
		}

		i++;
	}

	while (!stackEmpty(&stOp))
	{
		stackPush(st, stackTop(&stOp));
		stackPop(&stOp);
	}

	stackDestroy(&stOp);
}
