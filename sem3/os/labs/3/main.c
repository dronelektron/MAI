#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct _Node
{
	int value;
	struct _Node* bro;
	struct _Node* son;
} Node;

typedef struct _Data
{
	pthread_mutex_t mtx;
	Node* node;
	int value;
	int found;
} Data;

Node* treeCreateNode(int value);
void treeInsert(Node** root);
int treeFind(Node* root, int value);
void treeDestroy(Node** root);
void treePrint(Node* root, int level);

void *tFind(void* arg);
void createThread(pthread_t* t, Data* data, Node* root);

int main(void)
{
	int ch;
	int value;
	Node* root = NULL;

	while (1)
	{
		ch = getchar();

		if (ch == 'q')
			break;

		switch (ch)
		{
			case '+':
			{
				treeInsert(&root);

				break;
			}

			case 'p':
			{
				printf("====Tree:====\n");
				treePrint(root, 0);
				printf("=============\n");

				getchar();

				break;
			}

			case 'f':
			{
				getchar();
				scanf("%d", &value);
				getchar();

				if (treeFind(root, value))
					printf("Value %d found\n", value);
				else
					printf("Value %d not found\n", value);
				
				break;
			}

			case 'h':
			{
				printf("Commands:\n");
				printf("+ VALUE PATH/ - Insert value\n");
				printf("p - Print tree\n");
				printf("f VALUE - Find value\n");
				printf("h - Help\n");
				printf("q - Quit\n");

				getchar();

				break;
			}

			default:
			{
				getchar();

				printf("Error. Command not found\n");

				break;
			}
		}
	}

	treeDestroy(&root);

	return 0;
}

Node* treeCreateNode(int value)
{
	Node* node = (Node*)malloc(sizeof(Node));
	node->value = value;
	node->bro = NULL;
	node->son = NULL;

	return node;
}

void treeInsert(Node** root)
{
	int ch;
	int data = 0;
	int path = 0;
	
	getchar();

	while ((ch = getchar()) != ' ')
		data = data * 10 + ch - '0';
	
	while ((ch = getchar()) != '\n')
	{
		if (ch != '/')
			path = path * 10 + ch - '0';
		else
		{
			while (*root != NULL)
			{
				if ((*root)->value != path)
					root = &(*root)->bro;
				else
				{
					root = &(*root)->son;

					break;
				}
			}

			path = 0;
		}
	}

	while (*root != NULL)
		root = &(*root)->bro;

	*root = treeCreateNode(data);
}

int treeFind(Node* root, int value)
{
	pthread_t t;
	Data data;

	data.node = root;
	data.value = value;
	data.found = 0;

	pthread_mutex_init(&data.mtx, NULL);
	createThread(&t, &data, root);
	pthread_join(t, NULL);
	pthread_mutex_destroy(&data.mtx);

	return data.found;
}

void treeDestroy(Node** root)
{
	if (*root == NULL)
		return;

	treeDestroy(&(*root)->bro);
	treeDestroy(&(*root)->son);

	free(*root);

	*root = NULL;
}

void treePrint(Node* root, int level)
{
	if (root == NULL)
		return;

	printf("%*s%d\n", level * 4, "", root->value);

	treePrint(root->son, level + 1);
	treePrint(root->bro, level);
}

void *tFind(void* arg)
{
	Data* data = (Data*)arg;
	Node* node = data->node;
	int value = data->value;
	size_t threadsCnt = 0;
	size_t i = 0;
	pthread_t* threads = NULL;
	
	pthread_mutex_unlock(&data->mtx);

	while (node != NULL)
	{
		++threadsCnt;
		node = node->bro;
	}

	threads = (pthread_t*)malloc(sizeof(pthread_t) * threadsCnt);
	node = data->node;

	while (node != NULL)
	{
		if (node->value == value)
		{
			data->found = 1;

			return NULL;
		}

		if (node->son != NULL && data->found != 1)
			createThread(&threads[i++], data, node->son);

		node = node->bro;
	}
	
	for (i = 0; i < threadsCnt; ++i)
		pthread_join(threads[i], NULL);

	free(threads);

	pthread_exit(NULL);
}

void createThread(pthread_t* t, Data* data, Node* root)
{
	pthread_mutex_lock(&data->mtx);

	data->node = root;

	if (pthread_create(t, NULL, tFind, (void*)data) != 0)
		pthread_mutex_unlock(&data->mtx);
}
