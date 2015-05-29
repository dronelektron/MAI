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
	TBigInteger(const TBigInteger& bi);
	
	void Print() const;
	TBigInteger Abs() const;

	TBigInteger operator - () const;
	TBigInteger operator + (const TBigInteger& bi) const;
	TBigInteger operator - (const TBigInteger& bi) const;
	TBigInteger operator * (const TBigInteger& bi) const;
	TBigInteger operator / (const TBigInteger& bi) const;
	TBigInteger operator ^ (const TBigInteger& bi) const;
	
	bool operator < (const TBigInteger& bi) const;
	bool operator > (const TBigInteger& bi) const;
	bool operator == (const TBigInteger& bi) const;

private:
	int mSign;
	NDS::TVector<short> mNums;
	
	static const int mBase;

	void mFix();
	TBigInteger mPower(const TBigInteger& a, const TBigInteger& b) const;
};

#endif
