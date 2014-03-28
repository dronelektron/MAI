#ifndef BST_H
#define BST_H

#include <stdlib.h>

typedef struct _BstNode
{
	int _key;
	struct _BstNode *_left;
	struct _BstNode *_right;
} BstNode;

BstNode *bstInsert(BstNode **node, const int key);
BstNode *bstFind(BstNode **node, const int key);
void bstRemove(BstNode **node, const int key);
void bstDestroy(BstNode **node);

#endif
