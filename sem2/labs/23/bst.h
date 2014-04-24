#ifndef BST_H
#define BST_H

#include <stdlib.h>

typedef enum _kLetters
{
	A = 0, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
} kLetters;

typedef struct _BstNode
{
	kLetters _key;
	struct _BstNode *_left;
	struct _BstNode *_right;
} BstNode;

BstNode *bstInsert(BstNode **node, const kLetters key);
BstNode *bstFind(BstNode **node, const kLetters key);
int bstRemove(BstNode **node, const kLetters key);
void bstDestroy(BstNode **node);

#endif
