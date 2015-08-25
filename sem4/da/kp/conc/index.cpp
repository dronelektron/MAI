#include "index.h"

TIndex::TIndex() : STRUCT_POS_SIZE(sizeof(TUCHAR) + sizeof(TUINT) * 3 + sizeof(TUSHORT))
{
	SetDir(".");
}

void TIndex::SetDir(const std::string& dir)
{
	mDir = dir;
	mFileNameDict = mDir + "/dictionary";
	mFileNamePos = mDir + "/position";
	mFileNameFiles = mDir + "/files";
}

TIndex::TPosIndex::TPosIndex(TUINT beginPar, TUINT endPar)
{
	begin = beginPar;
	end = endPar;
}

TIndex::TPos::TPos(TUCHAR docIdPar, TUINT wordIdPar, TUINT offsetPar, TUINT rowPar, TUSHORT colPar)
{
	docId = docIdPar;
	wordId = wordIdPar;
	offset = offsetPar;
	row = rowPar;
	col = colPar;
}

bool TIndex::TPos::Less(const TPos& pos, int wordOffset) const
{
	if (docId == pos.docId)
	{
		int wordId1 = static_cast<int>(wordId);
		int wordId2 = static_cast<int>(pos.wordId);

		return wordId1 < wordId2 - wordOffset;
	}
	
	return docId < pos.docId;
}

bool TIndex::TPos::Equal(const TPos& pos, int wordOffset) const
{
	return !(Less(pos, wordOffset) || pos.Less(*this, -wordOffset));
}
