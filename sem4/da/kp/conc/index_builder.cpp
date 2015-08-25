#include "index_builder.h"

TIndexBuilder::TIndexBuilder() : MAX_POSITIONS_IN_BUFFER(10000000)
{
	mLastOffset = 0;
	mPosBufferSize = 0;
	mPosBuffer = new TPosBufferData[MAX_POSITIONS_IN_BUFFER];

	for (TUINT i = 0; i < MAX_POSITIONS_IN_BUFFER; ++i)
	{
		mPosBuffer[i].next = -1;
	}
}

TIndexBuilder::~TIndexBuilder()
{
	delete[] mPosBuffer;
}

void TIndexBuilder::Clear()
{
	FILE* fileTmp = fopen(mFileNamePos.c_str(), "wb");
	
	fclose(fileTmp);
}

std::pair<TUINT, TUINT> TIndexBuilder::Add(FILE* file, TUCHAR docId)
{
	std::pair<TUINT, TUINT> cnts;
	std::string word;
	TStringsSet terms;
	TTokenizer tok(file);

	cnts.first = 0;

	while (true)
	{
		tok.ReadNext(word);

		if (word.length() == 0)
		{
			break;
		}
		
		terms.insert(word);

		if (mDict.find(word) == mDict.end())
		{
			mDict.insert(TDictData(word, TPosIndex()));
		}

		mPosBuffer[mPosBufferSize].val = TPos(docId, tok.GetNum(), tok.GetOffset(), tok.GetRow(), tok.GetCol());

		if (mPosInfoDict.find(word) == mPosInfoDict.end())
		{
			mPosInfoDict.insert(TPosInfoDictData(word, TPosInfo(mPosBufferSize, mPosBufferSize)));
		}
		else
		{
			int& last = mPosInfoDict[word].second;

			mPosBuffer[last].next = mPosBufferSize;
			last = mPosBufferSize;
		}
		
		++mPosBufferSize;
		++cnts.first;
		
		if (mPosBufferSize == MAX_POSITIONS_IN_BUFFER)
		{
			mAddToIndex();
		}
	}
	
	mAddToIndex();

	cnts.second = terms.size();
	
	return cnts;
}

bool TIndexBuilder::Save(char** files, TUINT cnt)
{
	FILE* file = fopen(mFileNameFiles.c_str(), "wb");

	if (file == NULL)
	{
		return false;
	}
	
	fwrite(&cnt, sizeof(cnt), 1, file);

	for (TUINT i = 0; i < cnt; ++i)
	{
		TUINT len = strlen(files[i]) + 1;

		fwrite(&len, sizeof(len), 1, file);
		fwrite(files[i], sizeof(char), len, file);
	}

	fclose(file);

	file = fopen(mFileNameDict.c_str(), "wb");

	TUINT dictSize = mDict.size();

	fwrite(&dictSize, sizeof(dictSize), 1, file);

	for (TDict::iterator it = mDict.begin(); it != mDict.end(); ++it)
	{
		TUINT len = it->first.length() + 1;

		fwrite(&len, sizeof(len), 1, file);
		fwrite(it->first.c_str(), sizeof(char), len, file);
		fwrite(&it->second.begin, sizeof(it->second.begin), 1, file);
		fwrite(&it->second.end, sizeof(it->second.end), 1, file);
	}

	fclose(file);

	return true;
}

void TIndexBuilder::mAddToIndex()
{
	FILE* file = fopen(mFileNamePos.c_str(), "ab");

	for (TPosInfoDict::iterator it = mPosInfoDict.begin(); it != mPosInfoDict.end(); ++it)
	{
		TDict::iterator posIt = mDict.find(it->first);
		TUINT startOffset = mLastOffset;
		
		while (it->second.first != -1)
		{
			mWritePosToFile(mPosBuffer[it->second.first].val, file);
			
			int cur = it->second.first;

			it->second.first = mPosBuffer[cur].next;
			mPosBuffer[cur].next = -1;
			mLastOffset += STRUCT_POS_SIZE;
		}

		if (posIt->second.end == 0)
		{
			posIt->second.begin = startOffset;
		}
		else
		{
			fclose(file);

			file = fopen(mFileNamePos.c_str(), "rb+");

			fseek(file, posIt->second.end, SEEK_SET);

			mWritePosToFile(TPos(0, 0, startOffset, 1, 0), file);

			fclose(file);

			file = fopen(mFileNamePos.c_str(), "ab");
		}

		mWritePosToFile(TPos(), file);

		posIt->second.end = mLastOffset;
		mLastOffset += STRUCT_POS_SIZE;
	}

	fclose(file);

	mPosInfoDict.clear();
	mPosBufferSize = 0;
}

void TIndexBuilder::mWritePosToFile(const TPos& pos, FILE* file)
{
	fwrite(&pos.docId, sizeof(pos.docId), 1, file);
	fwrite(&pos.wordId, sizeof(pos.wordId), 1, file);
	fwrite(&pos.offset, sizeof(pos.offset), 1, file);
	fwrite(&pos.row, sizeof(pos.row), 1, file);
	fwrite(&pos.col, sizeof(pos.col), 1, file);
}
