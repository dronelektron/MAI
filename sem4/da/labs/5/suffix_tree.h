#ifndef SUFFIX_TREE_H
#define SUFFIX_TREE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "map.h"

class TST
{
public:
	struct TNode
	{
		int pat;
		int start;
		int end;
		TNode* slink;
		NDS::TMap<char, TNode*> next;
	};

	TST(const char* str);
	~TST();
	
	TNode* GetRoot();
	const char* GetText() const;

	//void Find(const char* str);
	//void Print();
	//void mPrint(TNode* node, int level);

private:
	TNode* mRoot;
	TNode* mNeedSL;
	TNode* mActiveNode;
	int mActiveE;
	int mActiveLen;
	int mRemainder;
	int mPos;
	int mINF;
	char* mText;

	TNode* mCreateNode(int start, int end);
	void mDestroyTree(TNode* node);
	int mEdgeLen(TNode* node);
	char mActiveEdge();
	bool mWalkDown(TNode* node);
	void mAddSL(TNode* node);
	void mExtend(int i);
	void mSuffNums(TNode* node, int len);

	//void mFindDFS(TNode* node, int deep);
};

#endif
