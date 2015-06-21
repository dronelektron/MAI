#include "suffix_tree.h"

TST::TST(const char* str)
{
	int len = strlen(str);

	mRoot = mCreateNode(-1, -1);
	mActiveNode = mRoot;
	mNeedSL = mRoot;
	mActiveE = 0;
	mActiveLen = 0;
	mRemainder = 0;
	mPos = -1;
	mINF = 1 << 30;
	mText = NULL;

	try
	{
		mText = new char[len + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");

		std::exit(0);
	}

	strcpy(mText, str);

	for (int i = 0; i < len; ++i)
	{
		mExtend(i);
	}

	mSuffNums(mRoot, 0);
}

TST::~TST()
{
	mDestroyTree(mRoot);

	delete[] mText;
}

TST::TNode* TST::GetRoot()
{
	return mRoot;
}

const char* TST::GetText() const
{
	return mText;
}
/*
void TST::Find(const char* str)
{
	int i = 0;
	int deep = 0;
	int len = strlen(str);
	TNode* node = mRoot;

	if (len > strlen(mText))
	{
		return;
	}

	while (i < len)
	{
		if (node->next.Find(str[i]) == NULL)
		{
			return;
		}

		deep += mEdgeLen(node);
		node = node->next.Find(str[i])->data.val;

		int edgeLen = mEdgeLen(node);
		int j = 0;

		while (j < edgeLen && i < len)
		{
			if (str[i] != mText[node->start + j])
			{
				return;
			}

			++i;
			++j;
		}
	}

	mFindDFS(node, deep);

	printf("\n");
}

void TST::Print()
{
	printf("--------\n");
	printf("Suffix Tree:\n");
	printf("--------\n");
	mPrint(mRoot, -1);
	printf("--------\n");
}

void TST::mPrint(TNode* node, int level)
{
	if (node != mRoot)
	{
		int len = strlen(mText);

		printf("%*s", level * 4, "");

		for (int i = node->start; i < node->end && i < len; ++i)
		{
			printf("%c", mText[i]);
		}

		printf("\t[%d, %d)\n", node->start, node->end);
	}

	for (NDS::TMap<char, TNode*>::TIterator it = node->next.Begin(); it != node->next.End(); ++it)
	{
		mPrint(it->val, level + 1);
	}
}
*/
TST::TNode* TST::mCreateNode(int start, int end)
{
	TNode* node = NULL;

	try
	{
		node = new TNode;
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");
		
		std::exit(0);
	}

	node->pat = -1;
	node->start = start;
	node->end = end;
	node->slink = NULL;
	
	return node;
}

void TST::mDestroyTree(TNode* node)
{
	for (NDS::TMap<char, TNode*>::TIterator it = node->next.Begin(); it != node->next.End(); ++it)
	{
		mDestroyTree(it->val);
	}

	delete node;
}

int TST::mEdgeLen(TNode* node)
{
	int a = node->end;
	int b = mPos + 1;

	return (a < b ? a : b) - node->start;
}

char TST::mActiveEdge()
{
	return mText[mActiveE];
}

bool TST::mWalkDown(TNode* node)
{
	if (mActiveLen >= mEdgeLen(node))
	{
		mActiveE += mEdgeLen(node);
		mActiveLen -= mEdgeLen(node);
		mActiveNode = node;

		return true;
	}

	return false;
}

void TST::mAddSL(TNode* node)
{
	if (mNeedSL != mRoot)
	{
		mNeedSL->slink = node;
	}

	mNeedSL = node;
}

void TST::mExtend(int i)
{
	mPos = i;
	mNeedSL = mRoot;
	++mRemainder;

	while (mRemainder > 0)
	{
		if (mActiveLen == 0)
		{
			mActiveE = mPos;
		}

		if (mActiveNode->next.Find(mActiveEdge()) == NULL)
		{
			TNode* leaf = mCreateNode(mPos, mINF);
			NDS::TMap<char, TNode*>::TData data;

			data.key = mActiveEdge();
			data.val = leaf;

			mActiveNode->next.Insert(data);

			mAddSL(mActiveNode);
		}
		else
		{
			TNode* nxt = mActiveNode->next.Find(mActiveEdge())->data.val;

			if (mWalkDown(nxt))
			{
				continue;
			}

			if (mText[nxt->start + mActiveLen] == mText[i])
			{
				++mActiveLen;

				mAddSL(mActiveNode);

				break;
			}

			TNode* split = mCreateNode(nxt->start, nxt->start + mActiveLen);
			TNode* leaf = mCreateNode(mPos, mINF);
			NDS::TMap<char, TNode*>::TData data;

			data.key = mActiveEdge();
			data.val = split;

			mActiveNode->next.Insert(data);

			data.key = mText[i];
			data.val = leaf;

			split->next.Insert(data);
			nxt->start += mActiveLen;

			data.key = mText[nxt->start];
			data.val = nxt;

			split->next.Insert(data);

			mAddSL(split);
		}

		--mRemainder;

		if (mActiveNode == mRoot && mActiveLen > 0)
		{
			--mActiveLen;
			mActiveE = mPos - mRemainder + 1;
		}
		else
		{
			mActiveNode = mActiveNode->slink != NULL ? mActiveNode->slink : mRoot;
		}
	}
}

void TST::mSuffNums(TNode* node, int len)
{
	if (node->end == mINF)
	{
		node->pat = node->start - len;
	}

	for (NDS::TMap<char, TNode*>::TIterator it = node->next.Begin(); it != node->next.End(); ++it)
	{
		mSuffNums(it->val, len + node->end - node->start);
	}
}
/*
void TST::mFindDFS(TNode* node, int deep)
{
	if (node->end == mINF)
	{
		printf(" %d", node->start - deep + 1);
	}

	for (NDS::TMap<char, TNode*>::TIterator it = node->next.Begin(); it != node->next.End(); ++it)
	{
		mFindDFS(it->val, deep + node->end - node->start);
	}
}
*/
