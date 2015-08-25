#ifndef INDEX_SEARCHER_H
#define INDEX_SEARCHER_H

#include <vector>
#include "index.h"

class TIndexSearcher : public TIndex
{
public:
	bool Load();
	bool GetContextWord(const std::string& query, TUINT cnt);
	bool GetContextSentence(const std::string& query, TUINT cnt);
	bool GetContextParagraph(const std::string& query, TUINT cnt);
	TUINT GetResultCount(const std::string& query);

private:
	typedef std::vector<std::string> TStringsVector;
	typedef std::vector<TUINT> TPosOffsetVector;
	typedef std::vector<TPos> TPosVector;

	class TIntersect
	{
	public:
		TIntersect(const TDict& dict, const TStringsVector& words, const std::string& fileName, TUINT jumpSize);
		~TIntersect();

		bool Get(TPos& posStart, TPos& posEnd);

	private:
		TPosOffsetVector mPosOffsets;
		TPosOffsetVector mPosOffsetsEnds;
		TUINT mSize;
		TUINT mJumpSize;
		FILE* mFile;

		bool mCheckIters() const;
		bool mEqual() const;
		void mNextPos(TUINT index);
		TPos mReadPosFromFile(TUINT index) const;
	};

	TDict mDict;
	TStringsVector mFileNames;

	TStringsVector mGetQueryWords(const std::string& query);

	static bool mIsSpace(int ch);
	static bool mIsStop(int ch);
	static bool mIsNewLine(int ch);

	std::string mFormat(FILE* file, TPos& posStart, TPos& posEnd, TUINT cnt, const std::string& delim, bool (*check)(int ch));
};

#endif
