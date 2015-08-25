#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <cstdio>
#include "types.h"

class TTokenizer
{
public:
	TTokenizer(FILE* file);

	void ReadNext(std::string& str);
	
	TUINT GetOffset() const;
	TUINT GetNum() const;
	TUINT GetRow() const;
	TUSHORT GetCol() const;

private:
	TUINT mOffset;
	TUINT mNum;
	TUINT mRow;
	TUSHORT mCol;
	TUSHORT mCurCol;
	TUINT mCurRow;
	FILE* mFile;
};

#endif
