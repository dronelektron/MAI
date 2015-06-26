#include "big_integer.h"

const short TBigInteger::mBASE = 10000;

TBigInteger::TBigInteger()
{
	mNums.PushBack(0);
}

TBigInteger::TBigInteger(int num)
{
	if (!num)
	{
		mNums.PushBack(0);
	}
	else
	{
		while (num)
		{
			mNums.PushBack(num % mBASE);
			num /= mBASE;
		}
	}
}

TBigInteger::TBigInteger(const char* str)
{
	int len = strlen(str);

	for (int i = len; i > 0; i -= 4)
	{
		short num = 0;

		for (int j = (i - 4 > 0 ? i - 4 : 0); j < i; ++j)
		{
			num = num * 10 + str[j] - '0';
		}

		mNums.PushBack(num);
	}

	mFix();
}

void TBigInteger::Print() const
{
	int size = mNums.Size();

	printf("%d", mNums[size - 1]);

	for (int i = size - 2; i >= 0; --i)
	{
		printf("%04d", mNums[i]);
	}

	printf("\n");
}

TBigInteger TBigInteger::operator + (const TBigInteger& bi) const
{
	short carry = 0;
	int maxSize = mNums.Size() > bi.mNums.Size() ? mNums.Size() : bi.mNums.Size();
	TBigInteger res;

	res.mNums.PopBack();

	for (int i = 0; i < maxSize; ++i)
	{
		short x = i < mNums.Size() ? mNums[i] : 0;
		short y = i < bi.mNums.Size() ? bi.mNums[i] : 0;
		short tmp = x + y + carry;

		carry = tmp / mBASE;
		res.mNums.PushBack(tmp % mBASE);
	}

	if (carry)
	{
		res.mNums.PushBack(carry);
	}

	return res;
}

TBigInteger TBigInteger::operator - (const TBigInteger& bi) const
{
	short carry = 0;
	TBigInteger res;

	res.mNums.PopBack();

	for (int i = 0; i < mNums.Size(); ++i)
	{
		short y = i < bi.mNums.Size() ? bi.mNums[i] : 0;
		short tmp = mNums[i] - y + carry;

		carry = tmp < 0 ? -1 : 0;
		res.mNums.PushBack(tmp < 0 ? tmp + mBASE : tmp);
	}

	res.mFix();

	return res;
}

TBigInteger TBigInteger::operator * (const TBigInteger& bi) const
{
	int m = mNums.Size();
	int n = bi.mNums.Size();
	int maxSize = m > n ? m : n;
	TBigInteger res;

	res.mNums.Resize(maxSize * 2, 0);

	for (int j = 0; j < n; ++j)
	{
		if (bi.mNums[j] != 0)
		{
			short carry = 0;

			for (int i = 0; i < m; ++i)
			{
				int tmp = mNums[i] * bi.mNums[j] + res.mNums[i + j] + carry;

				carry = tmp / mBASE;
				res.mNums[i + j] = tmp % mBASE;
			}

			res.mNums[j + m] = carry;
		}
	}

	res.mFix();

	return res;
}

TBigInteger TBigInteger::operator / (const TBigInteger& bi) const
{
	int norm = mBASE / (bi.mNums.Back() + 1);

	TBigInteger a = *this * norm;
	TBigInteger b = bi * norm;
	TBigInteger q;
	TBigInteger r;

	q.mNums.Resize(a.mNums.Size());

	for (int i = a.mNums.Size() - 1; i >= 0; --i)
	{
		r = r * mBASE;
		r = r + a.mNums[i];

		short s1 = r.mNums.Size() <= b.mNums.Size() ? 0 : r.mNums[b.mNums.Size()];
		short s2 = r.mNums.Size() <= b.mNums.Size() - 1 ? 0 : r.mNums[b.mNums.Size() - 1];
		short d = static_cast<short>((static_cast<int>(s1) * mBASE + s2) / b.mNums.Back());
		
		TBigInteger tmp = b * d;

		while (tmp > r)
		{
			tmp = tmp - b;
			--d;
		}

		r = r - tmp;
		q.mNums[i] = d;
	}

	q.mFix();

	return q;
}

TBigInteger TBigInteger::operator ^ (const TBigInteger& bi) const
{
	return mPower(bi);
}

bool TBigInteger::operator < (const TBigInteger& bi) const
{
	if (mNums.Size() != bi.mNums.Size())
	{
		return mNums.Size() < bi.mNums.Size();
	}

	for (int i = mNums.Size() - 1; i >= 0; --i)
	{
		if (mNums[i] != bi.mNums[i])
		{
			return mNums[i] < bi.mNums[i];
		}
	}

	return false;
}

bool TBigInteger::operator > (const TBigInteger& bi) const
{
	return bi < *this;
}

bool TBigInteger::operator == (const TBigInteger& bi) const
{
	return !(*this < bi) && !(bi < *this);
}

void TBigInteger::mFix()
{
	while (mNums.Size() > 1 && mNums.Back() == 0)
	{
		mNums.PopBack();
	}
}

TBigInteger TBigInteger::mPower(const TBigInteger& b) const
{
	if (b == 0)
	{
		return 1;
	}

	if (b.mNums[0] & 1)
	{
		return *this * mPower(b - 1);
	}
	
	TBigInteger c = mPower(b / 2);

	return c * c;
}
