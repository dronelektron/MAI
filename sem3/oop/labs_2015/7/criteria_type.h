#ifndef CRITERIA_TYPE_H
#define CRITERIA_TYPE_H

#include <cstring>
#include "criteria.h"

template <class T>
class CriteriaType : public Criteria<T>
{
public:
	CriteriaType(const char* type);

	bool check(const std::shared_ptr<T>& item) const override;

private:
	char m_type[16];
};

#include "criteria_type_impl.cpp"

#endif
