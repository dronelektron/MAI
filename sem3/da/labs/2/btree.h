#ifndef TBTREE_H
#define TBTREE_H

#include <exception>
#include <new>
#include <cstdlib>
#include <cstdio>
#include "string.h"

typedef unsigned long long TULL;

class TBTree
{
public:
	typedef TString TKey;
	typedef TULL TVal;

	struct TData
	{
		TKey key;
		TVal val;
	};

	struct TNode
	{
		bool leaf;
		size_t n;
		TData* data;
		TNode** childs;
	};

	explicit TBTree(size_t t);
	~TBTree();

	bool Insert(const TData& data);
	TData* Find(const TKey& key);
	void Erase(const TKey& key);

	bool Serialize(const char* filename);
	bool Deserialize(const char* filename);

private:
	TNode* mRoot;
	size_t mT;

	TNode* mCreateNode();
	TData* mFind(TNode* node, const TKey& key);
	void mDeleteNode(TNode* node);
	void mDeleteTree(TNode* node);
	void mShiftLeft(TNode* node, size_t index, size_t index2);
	void mShiftRight(TNode* node, size_t index, size_t index2);
	void mFlowLeft(TNode* parent, size_t index);
	void mFlowRight(TNode* parent, size_t index);
	void mSplitNode(TNode* parent, TNode* node, size_t index);
	void mMergeNode(TNode* parent, size_t index);
	bool mInsertNonFull(TNode* node, const TData& data);
	void mErase(TNode* node, const TKey& key);
	void mFixChild(TNode* node, size_t index);
	bool mSerialize(FILE* f, TNode* node);
	bool mDeserialize(FILE* f, TNode* node);
};

#endif
