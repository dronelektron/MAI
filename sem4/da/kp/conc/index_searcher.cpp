#include "index_searcher.h"

bool TIndexSearcher::Load()
{
	TUINT cnt = 0;
	FILE* file = fopen(mFileNameFiles.c_str(), "rb");

	if (file == NULL)
	{
		return false;
	}

	fread(&cnt, sizeof(cnt), 1, file);

	mFileNames.resize(cnt);

	for (TUINT i = 0; i < cnt; ++i)
	{
		TUINT len = 0;

		fread(&len, sizeof(len), 1, file);

		char* str = new char[len];

		fread(str, sizeof(char), len, file);

		mFileNames[i] = str;

		delete[] str;
	}

	fclose(file);

	file = fopen(mFileNameDict.c_str(), "rb");

	fread(&cnt, sizeof(cnt), 1, file);

	for (TUINT i = 0; i < cnt; ++i)
	{
		TUINT len = 0;

		fread(&len, sizeof(len), 1, file);

		char* str = new char[len];

		fread(str, sizeof(char), len, file);

		TPosIndex& posInd = mDict[str];

		fread(&posInd.begin, sizeof(posInd.begin), 1, file);
		fread(&posInd.end, sizeof(posInd.end), 1, file);
	}

	fclose(file);
	
	return true;
}

bool TIndexSearcher::GetContextWord(const std::string& query, TUINT cnt)
{
	TStringsVector words = mGetQueryWords(query);

	if (words.size() == 0)
	{
		return false;
	}

	bool isFound = false;
	TIntersect intersect(mDict, words, mFileNamePos, STRUCT_POS_SIZE);
	TPos posStart;
	TPos posEnd;

	while (intersect.Get(posStart, posEnd))
	{
		printf("! File %s, line %u, position %hu:\n", mFileNames[posStart.docId].c_str(), posStart.row, posStart.col);

		// CONTEXT
		
		FILE* file = fopen(mFileNames[posStart.docId].c_str(), "r");

		printf("================================\n");
		printf("%s", mFormat(file, posStart, posEnd, cnt, " ", mIsSpace).c_str());
		printf("================================\n");

		fclose(file);

		// END CONTEXT

		isFound = true;
	}

	return isFound;
}

bool TIndexSearcher::GetContextSentence(const std::string& query, TUINT cnt)
{
	TStringsVector words = mGetQueryWords(query);

	if (words.size() == 0)
	{
		return false;
	}

	bool isFound = false;
	TIntersect intersect(mDict, words, mFileNamePos, STRUCT_POS_SIZE);
	TPos posStart;
	TPos posEnd;

	while (intersect.Get(posStart, posEnd))
	{
		printf("! File %s, line %u, position %hu:\n", mFileNames[posStart.docId].c_str(), posStart.row, posStart.col);

		// CONTEXT START
		
		FILE* file = fopen(mFileNames[posStart.docId].c_str(), "r");

		printf("================================\n");
		printf("%s", mFormat(file, posStart, posEnd, cnt, "\n", mIsStop).c_str());
		printf("================================\n");

		fclose(file);

		// CONTEXT END
		
		isFound = true;
	}

	return isFound;
}

bool TIndexSearcher::GetContextParagraph(const std::string& query, TUINT cnt)
{
	TStringsVector words = mGetQueryWords(query);

	if (words.size() == 0)
	{
		return false;
	}

	bool isFound = false;
	TIntersect intersect(mDict, words, mFileNamePos, STRUCT_POS_SIZE);
	TPos posStart;
	TPos posEnd;

	while (intersect.Get(posStart, posEnd))
	{
		printf("! File %s, line %u, position %hu:\n", mFileNames[posStart.docId].c_str(), posStart.row, posStart.col);

		// CONTEXT START

		FILE* file = fopen(mFileNames[posStart.docId].c_str(), "r");

		printf("================================\n");
		printf("%s", mFormat(file, posStart, posEnd, cnt, "\n\n", mIsNewLine).c_str());
		printf("================================\n");

		fclose(file);

		// CONTEXT END

		isFound = true;
	}

	return isFound;
}

TUINT TIndexSearcher::GetResultCount(const std::string& query)
{
	TUINT cnt = 0;
	TStringsVector words = mGetQueryWords(query);

	if (words.size() == 0)
	{
		return false;
	}

	TIntersect intersect(mDict, words, mFileNamePos, STRUCT_POS_SIZE);
	TPos posStart;
	TPos posEnd;

	while (intersect.Get(posStart, posEnd))
	{
		++cnt;
	}

	return cnt;
}

TIndexSearcher::TIntersect::TIntersect(const TDict& dict, const TStringsVector& words, const std::string& fileName, TUINT jumpSize)
{
	mPosOffsets.resize(words.size());
	mPosOffsetsEnds.resize(words.size());
	mSize = words.size();
	mJumpSize = jumpSize;
	mFile = fopen(fileName.c_str(), "rb");

	for (TUINT i = 0; i < mSize; ++i)
	{
		TDict::const_iterator it = dict.find(words[i]);

		mPosOffsets[i] = it->second.begin;
		mPosOffsetsEnds[i] = it->second.end;
	}
}

TIndexSearcher::TIntersect::~TIntersect()
{
	fclose(mFile);
}

bool TIndexSearcher::TIntersect::Get(TPos& posStart, TPos& posEnd)
{
	while (mCheckIters())
	{
		for (TUINT i = 0; i < mSize; ++i)
		{
			TPos pos = mReadPosFromFile(i);

			if (pos.wordId == 0 && pos.row == 1)
			{
				mNextPos(i);
			}
		}

		if (mEqual())
		{
			posStart = mReadPosFromFile(0);
			posEnd = mReadPosFromFile(mSize - 1);

			for (TUINT i = 0; i < mSize; ++i)
			{
				mNextPos(i);
			}

			return true;
		}

		TPos maxPos = mReadPosFromFile(0);
		TUINT maxInd = 0;

		for (TUINT i = 1; i < mSize; ++i)
		{
			TPos pos = mReadPosFromFile(i);

			if (maxPos.Less(pos, 0))
			{
				maxPos = pos;
				maxInd = i;
			}
		}

		for (TUINT i = 0; i < mSize; ++i)
		{
			if (i == maxInd)
			{
				continue;
			}

			while (mPosOffsets[i] != mPosOffsetsEnds[i] && mReadPosFromFile(i).Less(maxPos, maxInd - i))
			{
				mNextPos(i);
			}
		}
	}
	
	return false;
}

bool TIndexSearcher::TIntersect::mCheckIters() const
{
	for (TUINT i = 0; i < mSize; ++i)
	{
		if (mPosOffsets[i] == mPosOffsetsEnds[i])
		{
			return false;
		}
	}
	
	return true;
}

bool TIndexSearcher::TIntersect::mEqual() const
{
	TUINT matched = 0;
	TPos firstPos = mReadPosFromFile(0);
	
	for (TUINT i = 0; i < mSize; ++i)
	{
		if (firstPos.Equal(mReadPosFromFile(i), i))
		{
			++matched;
		}
	}

	return matched == mSize;
}

void TIndexSearcher::TIntersect::mNextPos(TUINT index)
{
	TPos pos = mReadPosFromFile(index);
	
	if (pos.wordId == 0 && pos.row == 1)
	{
		mPosOffsets[index] = pos.offset;
	}
	else
	{
		mPosOffsets[index] += mJumpSize;
	}
}

TIndex::TPos TIndexSearcher::TIntersect::mReadPosFromFile(TUINT index) const
{
	TPos pos;

	fseek(mFile, mPosOffsets[index], SEEK_SET);
	fread(&pos.docId, sizeof(pos.docId), 1, mFile);
	fread(&pos.wordId, sizeof(pos.wordId), 1, mFile);
	fread(&pos.offset, sizeof(pos.offset), 1, mFile);
	fread(&pos.row, sizeof(pos.row), 1, mFile);
	fread(&pos.col, sizeof(pos.col), 1, mFile);

	return pos;
}

TIndexSearcher::TStringsVector TIndexSearcher::mGetQueryWords(const std::string& query)
{
	TStringsVector words;
	std::string word;

	for (TUINT i = 0; i <= query.length(); ++i)
	{
		if (isalpha(query[i]))
		{
			word += static_cast<char>(tolower(query[i]));
		}
		else if (word.length() > 0)
		{
			if (mDict.find(word) == mDict.end())
			{
				words.clear();

				return words;
			}

			words.push_back(word);
			word = "";
		}
	}

	return words;
}

bool TIndexSearcher::mIsSpace(int ch)
{
	return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

bool TIndexSearcher::mIsStop(int ch)
{
	return ch == '.' || ch == '?' || ch == '!';
}

bool TIndexSearcher::mIsNewLine(int ch)
{
	return ch == '\n';
}

std::string TIndexSearcher::mFormat(FILE* file, TPos& posStart, TPos& posEnd, TUINT cnt, const std::string& delim, bool (*check)(int ch))
{
	int ch;
	bool isFound = false;
	std::string res;
	TUINT curCnt = 0;
	TUINT offset1 = posStart.offset;
	TUINT offset2 = posEnd.offset;

	while (offset1 > 0)
	{
		--offset1;
		
		fseek(file, offset1, SEEK_SET);

		ch = fgetc(file);

		if (check(ch))
		{
			break;
		}
	}

	while (offset1 > 0 && curCnt < cnt)
	{
		--offset1;

		fseek(file, offset1, SEEK_SET);

		ch = fgetc(file);

		if (!check(ch))
		{
			isFound = true;
		}
		else if (isFound)
		{
			++curCnt;
			isFound = false;
		}
	}

	fseek(file, offset2, SEEK_SET);

	while ((ch = fgetc(file)) != EOF)
	{
		++offset2;

		if (check(ch))
		{
			break;
		}
	}

	curCnt = 0;

	while ((ch = fgetc(file)) != EOF && curCnt < cnt)
	{
		++offset2;

		if (!check(ch))
		{
			isFound = true;
		}
		else if (isFound)
		{
			++curCnt;
			isFound = false;
		}
	}

	fseek(file, offset1, SEEK_SET);

	while (offset1 < offset2)
	{
		++offset1;

		ch = fgetc(file);

		if (mIsSpace(res[res.length() - 1]) && mIsSpace(ch))
		{
			continue;
		}
		
		res += mIsSpace(ch) ? ' ' : ch;

		if (check(ch))
		{
			res += delim;
		}
	}

	TUINT startLeft = 0;

	while (startLeft < res.length() - 1 && (mIsSpace(res[startLeft]) || mIsStop(res[startLeft]) || mIsNewLine(res[startLeft])))
	{
		++startLeft;
	}

	if (res[res.length() - 1] != '\n')
	{
		res += '\n';
	}
	else if (check != mIsStop)
	{
		res.resize(res.length() - 1);
	}

	return &res.c_str()[startLeft];
}
