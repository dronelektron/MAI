#include "suffix_array.h"

TSA::TSA(TST& tst)
{
	const char* text = tst.GetText();

	mTextLen = strlen(text);
	mText = NULL;
	mArr = NULL;
	mCurInd = 0;

	try
	{
		mText = new char[mTextLen + 1];
		mArr = new int[mTextLen];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");
		
		std::exit(0);
	}

	strcpy(mText, text);
	mBuild(tst.GetRoot());
}

TSA::~TSA()
{
	delete[] mText;
	delete[] mArr;
}

void TSA::Find(const char* str)
{
	int left = mFindLeft(str);
	int right = mFindRight(str);
	int size = 0;
	int cap = 1;
	int* res = NULL;

	try
	{
		res = new int[cap];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");
		
		std::exit(0);
	}

	for (int i = left; i < right; ++i)
	{
		if (size == cap)
		{
			cap *= 2;

			int* arr = NULL;

			try
			{
				arr = new int[cap];
			}
			catch (const std::bad_alloc& e)
			{
				printf("No memory\n");
				
				std::exit(0);
			}

			for (int j = 0; j < size; ++j)
			{
				arr[j] = res[j];
			}

			delete[] res;

			res = arr;
		}

		res[size++] = mArr[i] + 1;
	}

	mRadixSort(res, size);

	if (left < right)
	{
		printf("%d", res[0]);

		for (size_t i = 1; i < size; ++i)
		{
			printf(", %d", res[i]);
		}
	}

	printf("\n");
	
	delete[] res;
}

int TSA::mFindLeft(const char* str) const
{
	int left = 0;
	int right = mTextLen - 1;
	int len = strlen(str);

	while (left <= right)
	{
		int mid = (left + right) / 2;

		if (strncmp(str, mText + mArr[mid], len) > 0)
		{
			left = mid + 1;
		}
		else
		{
			right = mid - 1;
		}
	}

	return left;
}

int TSA::mFindRight(const char* str) const
{
	int left = 0;
	int right = mTextLen - 1;
	int len = strlen(str);

	while (left <= right)
	{
		int mid = (left + right) / 2;

		if (strncmp(str, mText + mArr[mid], len) >= 0)
		{
			left = mid + 1;
		}
		else
		{
			right = mid - 1;
		}
	}
	
	return left;
}

void TSA::mBuild(TST::TNode* node)
{
	for (NDS::TMap<char, TST::TNode*>::TIterator it = node->next.Begin(); it != node->next.End(); ++it)
	{
		mBuild(it->val);
	}
	
	if (node->end > mTextLen)
	{
		mArr[mCurInd++] = node->pat;
	}
}

void TSA::mRadixSort(int*& arr, int size)
{
	if (size < 2)
	{
		return;
	}

	const int MAX = 255;

	int* cnts = NULL;
	int* arr2 = NULL;
	
	try
	{
		cnts = new int[MAX + 1];
		arr2 = new int[size];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");
		
		std::exit(0);
	}

	for (int i = 0; i < sizeof(int); ++i)
	{
		for (int j = 0; j <= MAX; ++j)
		{
			cnts[j] = 0;
		}

		for (int j = 0; j < size; ++j)
		{
			++cnts[(arr[j] >> (i * 8)) & MAX];
		}

		for (int j = 1; j <= MAX; ++j)
		{
			cnts[j] += cnts[j - 1];
		}

		for (int j = size - 1; j >= 0; --j)
		{
			arr2[--cnts[(arr[j] >> (i * 8)) & MAX]] = arr[j];
		}

		int* ptr = arr;

		arr = arr2;
		arr2 = ptr;
	}

	delete[] cnts;
	delete[] arr2;
}
