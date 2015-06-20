#ifndef SUFFIX_ARRAY_H
#define SUFFIX_ARRAY_H

#include "suffix_tree.h"

class TSA
{
public:
	TSA(TST& tst);
	~TSA();

	void Find(const char* str);

private:
	char* mText;
	int* mArr;
	int mCurInd;
	int mTextLen;

	void mBuild(TST::TNode* node);
	int mFindLeft(const char* str) const;
	int mFindRight(const char* str) const;

	void mRadixSort(int*& arr, int size);
};

#endif
