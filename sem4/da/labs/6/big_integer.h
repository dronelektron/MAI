#ifndef BIG_INTEGER_H
#define BIG_INTEGER_H

#include <cmath>
#include <cstring>
#include <cstdio>
#include "vector.h"

class TBigInteger
{
public:
	TBigInteger();
	TBigInteger(int num);
	TBigInteger(const char* str);
	
	void Print() const;
	
	TBigInteger operator + (const TBigInteger& bi) const;
	TBigInteger operator - (const TBigInteger& bi) const;
	TBigInteger operator * (const TBigInteger& bi) const;
	TBigInteger operator / (const TBigInteger& bi) const;
	TBigInteger operator ^ (const TBigInteger& bi) const;
	
	bool operator < (const TBigInteger& bi) const;
	bool operator > (const TBigInteger& bi) const;
	bool operator == (const TBigInteger& bi) const;

private:
	NDS::TVector<short> mNums;
	
	static const int mBASE;

	void mFix();
	TBigInteger mPower(const TBigInteger& bi) const;
};

#endif
