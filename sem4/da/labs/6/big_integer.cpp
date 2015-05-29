#include "big_integer.h"

const int TBigInteger::mBase = 10000;

TBigInteger::TBigInteger()
{
	mSign = 1;
	mNums.PushBack(0);
}

TBigInteger::TBigInteger(int num)
{
	mSign = 1;

	if (!num)
	{
		mNums.PushBack(0);
	}
	else
	{
		if (num < 0)
		{
			mSign = -1;
			num = -num;
		}

		while (num)
		{
			mNums.PushBack(num % mBase);
			num /= mBase;
		}
	}
}

TBigInteger::TBigInteger(const char* str)
{
	mSign = 1;

	int len = strlen(str);

	if (!len)
	{
		mNums.PushBack(0);
	}
	else
	{
		for (int i = len - 1; i >= 0; i -= 4)
		{
			int k = 1;
			int num = 0;

			for (int j = i; j >= 0 && j >= i - 3; --j)
			{
				num += (str[j] - '0') * k;
				k *= 10;
			}

			mNums.PushBack(num);
		}
	}

	mFix();
}

TBigInteger::TBigInteger(const TBigInteger& bi)
{
	mSign = bi.mSign;
	mNums = bi.mNums;
}

void TBigInteger::Print() const
{
	int size = mNums.Size();

	if (mSign == -1)
	{
		printf("-");
	}

	printf("%d", mNums[size - 1]);

	for (int i = size - 2; i >= 0; --i)
	{
		printf("%04d", mNums[i]);
	}

	printf("\n");
}

TBigInteger TBigInteger::Abs() const
{
	TBigInteger res(*this);

	res.mSign *= res.mSign;

	return res;
}

TBigInteger TBigInteger::operator - () const
{
	TBigInteger res(*this);

	res.mSign = -mSign;

	return res;
}

TBigInteger TBigInteger::operator + (const TBigInteger& bi) const
{
	if (mSign != bi.mSign)
	{
		return *this - (-bi);
	}

	int carry = 0;
	int size1 = mNums.Size();
	int size2 = bi.mNums.Size();
	int minSize = size1 < size2 ? size1 : size2;
	int maxSize = size1 > size2 ? size1 : size2;
	TBigInteger res;

	res.mSign = mSign;

	const TBigInteger* biMax = size1 > size2 ? this : &bi;

	res.mNums.PopBack();
	
	for (size_t i = 0; i < minSize; ++i)
	{
		int t = mNums[i] + bi.mNums[i] + carry;

		carry = t / mBase;
		res.mNums.PushBack(t % mBase);
	}

	for (size_t i = minSize; i < maxSize; ++i)
	{
		int t = biMax->mNums[i] + carry;

		carry = t / mBase;
		res.mNums.PushBack(t % mBase);
	}
	
	if (carry)
	{
		res.mNums.PushBack(carry);
	}

	return res;
}

TBigInteger TBigInteger::operator - (const TBigInteger& bi) const
{
	if (mSign == bi.mSign)
	{
		if (!(Abs() < bi.Abs()))
		{
			int carry = 0;
			TBigInteger res;

			res.mSign = mSign;
			res.mNums.PopBack();
			
			for (size_t i = 0; i < bi.mNums.Size(); ++i)
			{
				int t = mNums[i] - bi.mNums[i] + carry;
				int w = (t < 0 ? mBase + t : t) % mBase;

				carry = static_cast<int>(floor(static_cast<double>(t) / mBase));
				res.mNums.PushBack(w);
			}
			
			for (size_t i = bi.mNums.Size(); i < mNums.Size(); ++i)
			{
				int t = mNums[i] + carry;
				int w = (t < 0 ? mBase + t : t) % mBase;

				carry = static_cast<int>(floor(static_cast<double>(t) / mBase));
				res.mNums.PushBack(w);
			}

			res.mFix();

			return res;
		}

		return -(bi - *this);
	}

	return *this + (-bi);
}

TBigInteger TBigInteger::operator * (const TBigInteger& bi) const
{
	int m = mNums.Size();
	int n = bi.mNums.Size();
	int sizeMax = m > n ? m : n;
	TBigInteger res;

	res.mNums.Resize(sizeMax * 2, 0);

	for (int j = 0; j < n; ++j)
	{
		if (bi.mNums[j] != 0)
		{
			int carry = 0;

			for (int i = 0; i < m; ++i)
			{
				int t = mNums[i] * bi.mNums[j] + res.mNums[i + j] + carry;

				res.mNums[i + j] = t % mBase;
				carry = static_cast<int>(floor(static_cast<double>(t) / mBase));
			}

			res.mNums[j + m] = carry;
		}
	}

	res.mFix();
	
	return res;
}

TBigInteger TBigInteger::operator / (const TBigInteger& bi) const
{
	int norm = mBase / (bi.mNums.Back() + 1);

	TBigInteger a = Abs() * norm;
	TBigInteger b = bi.Abs() * norm;
	TBigInteger q;
	TBigInteger r;

	q.mNums.Resize(a.mNums.Size());

	for (int i = a.mNums.Size() - 1; i >= 0; --i)
	{
		r = r * mBase;
		r = r + a.mNums[i];

		int s1 = r.mNums.Size() <= b.mNums.Size() ? 0 : r.mNums[b.mNums.Size()];
		int s2 = r.mNums.Size() <= b.mNums.Size() - 1 ? 0 : r.mNums[b.mNums.Size() - 1];
		int d = (static_cast<long long>(mBase) * s1 + s2) / b.mNums.Back();

		r = r - b * d;
		
		while (r < 0)
		{
			r = r + b;
			--d;
		}
		
		q.mNums[i] = d;
	}

	q.mSign = mSign * bi.mSign;
	q.mFix();

	return q;
}

TBigInteger TBigInteger::operator ^ (const TBigInteger& bi) const
{
	return mPower(*this, bi);
}

bool TBigInteger::operator < (const TBigInteger& bi) const
{
	if (mSign != bi.mSign)
	{
		return mSign < bi.mSign;
	}

	if (mNums.Size() != bi.mNums.Size())
	{
		return mNums.Size() * mSign < bi.mNums.Size() * bi.mSign;
	}

	if (mNums.Size() < bi.mNums.Size())
	{
		return true;
	}

	for (int i = mNums.Size() - 1; i >= 0; --i)
	{
		if (mNums[i] != bi.mNums[i])
		{
			return mNums[i] * mSign < bi.mNums[i] * bi.mSign;
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
	
	if (mNums.Size() == 1 && mNums[0] == 0)
	{
		mSign = 1;
	}
}

TBigInteger TBigInteger::mPower(const TBigInteger& a, const TBigInteger& b) const
{
	if (b == 0)
	{
		return 1;
	}

	if (b.mNums[0] & 1)
	{
		return a * mPower(a, b - 1);
	}
	
	TBigInteger c = mPower(a, b / 2);

	return c * c;
}
