#ifndef INDEX_BUILDER_H
#define INDEX_BUILDER_H

#include <set>
#include "index.h"
#include "tokenizer.h"

class TIndexBuilder : public TIndex
{
public:
	TIndexBuilder();
	~TIndexBuilder();
	
	void Clear();
	std::pair<TUINT, TUINT> Add(FILE* file, TUCHAR docId);
	bool Save(char** files, TUINT cnt);

private:
	typedef std::pair<std::string, TPosIndex> TDictData;
	typedef std::pair<int, int> TPosInfo;
	typedef std::pair<std::string, TPosInfo> TPosInfoDictData;
	typedef std::set<std::string> TStringsSet;
	typedef std::map<std::string, TPosInfo> TPosInfoDict;

	struct TPosBufferData
	{
		TPos val;
		int next;
	};

	TDict mDict;
	TUINT mLastOffset;
	TUINT mPosBufferSize;
	TPosBufferData* mPosBuffer;
	TPosInfoDict mPosInfoDict;

	const TUINT MAX_POSITIONS_IN_BUFFER;
	
	void mAddToIndex();
	void mWritePosToFile(const TPos& pos, FILE* file);
};

#endif
