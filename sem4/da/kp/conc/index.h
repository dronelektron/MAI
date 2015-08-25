#ifndef INDEX_H
#define INDEX_H

#include <map>
#include <string>
#include <cstdio>
#include <cctype>
#include <cstring>
#include "types.h"

class TIndex
{
public:
	TIndex();

	void SetDir(const std::string& dir);

protected:
	struct TPosIndex
	{
		TPosIndex(TUINT startPar = 0, TUINT endPar = 0);

		TUINT begin;
		TUINT end;
	};

	struct TPos
	{
		TPos(TUCHAR docIdPar = 0, TUINT wordIdPar = 0, TUINT offsetPar = 0, TUINT rowPar = 0, TUSHORT colPar = 0);
		
		TUCHAR docId;
		TUINT wordId;
		TUINT offset;
		TUINT row;
		TUSHORT col;
		
		bool Less(const TPos& pos, int wordOffset) const;
		bool Equal(const TPos& pos, int wordOffset) const;
	};
	
	typedef std::map<std::string, TPosIndex> TDict;
	
	std::string mDir;
	std::string mFileNameDict;
	std::string mFileNamePos;
	std::string mFileNameFiles;

	const TUINT STRUCT_POS_SIZE;
};

#endif
