#ifndef BST_H
#define BST_H

#include <stdlib.h>

typedef int BST_TYPE;

typedef struct _BstNode
{
	BST_TYPE key;
	struct _BstNode *left;
	struct _BstNode *right;
} BstNode;

BstNode *bstInsert(BstNode **node, const BST_TYPE key);
BstNode *bstFind(BstNode **node, const BST_TYPE key);
void bstRemove(BstNode **node, const BST_TYPE key);
void bstDestroy(BstNode **node);

#endif
