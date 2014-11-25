#include "string.h"

TString::TString()
{
	try
	{
		mStr = new char[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	mStr[0] = '\0';
}

TString::TString(const char* s)
{
	try
	{
		mStr = new char[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	mStr[0] = '\0';

	mCopy2(s);
}

TString::~TString()
{
	delete[] mStr;
}

size_t TString::Length() const
{
	return strlen(mStr);
}

const char* TString::Str() const
{
	return mStr;
}

void TString::Swap(TString& s)
{
	char* buffer = mStr;

	mStr = s.mStr;
	s.mStr = buffer;
}

TString& TString::operator=(const TString& s)
{
	if (this != &s)
	{
		mCopy(s);
	}

	return *this;
}

bool TString::operator<(const TString& s) const
{
	return strcmp(mStr, s.mStr) < 0;
}

bool TString::operator==(const TString& s) const
{
	return strcmp(mStr, s.mStr) == 0;
}

void TString::mCopy(const TString& s)
{
	delete[] mStr;

	size_t n = s.Length();

	try
	{
		mStr = new char[n + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	strcpy(mStr, s.mStr);
}

void TString::mCopy2(const char* s)
{
	delete[] mStr;

	size_t n = strlen(s);

	try
	{
		mStr = new char[n + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}
	
	strcpy(mStr, s);
}
