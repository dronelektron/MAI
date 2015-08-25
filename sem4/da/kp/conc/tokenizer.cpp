#include "tokenizer.h"

TTokenizer::TTokenizer(FILE* file)
{
	mOffset = 0;
	mNum = 0;
	mRow = 0;
	mCol = 0;
	mCurRow = 1;
	mCurCol = 0;
	mFile = file;
}

void TTokenizer::ReadNext(std::string& str)
{
	int ch;
	bool isFound = false;
	
	str = "";
	
	while ((ch = fgetc(mFile)) != EOF)
	{
		++mCurCol;

		if (ch == EOF)
		{
			break;
		}
		else if (isalpha(ch))
		{
			if (!isFound)
			{
				isFound = true;
				mOffset = ftell(mFile) - 1;
				mRow = mCurRow;
				mCol = mCurCol;
				++mNum;
			}

			str += static_cast<char>(tolower(ch));
		}
		else
		{
			if (ch == '\n')
			{
				++mCurRow;
				mCurCol = 0;
			}

			if (isFound)
			{
				break;
			}
		}
	}
}

TUINT TTokenizer::GetOffset() const
{
	return mOffset;
}

TUINT TTokenizer::GetNum() const
{
	return mNum;
}

TUINT TTokenizer::GetRow() const
{
	return mRow;
}

TUSHORT TTokenizer::GetCol() const
{
	return mCol;
}
