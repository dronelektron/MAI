#ifndef TSTRING_H
#define TSTRING_H

#include <exception>
#include <new>
#include <cstdlib>
#include <cstring>
#include <cstdio>

class TString
{
public:
	TString();
	TString(const char* s);
	~TString();

	size_t Length() const;
	const char* Str() const;
	void Swap(TString& s);

	TString& operator=(const TString& s);
	bool operator<(const TString& s) const;
	bool operator==(const TString& s) const;

private:
	char* mStr;

	void mCopy(const TString& s);
	void mCopy2(const char* s);
};

#endif
