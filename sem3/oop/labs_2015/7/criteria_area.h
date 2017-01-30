#ifndef CRITERIA_AREA_H
#define CRITERIA_AREA_H

#include "criteria.h"

template <class T>
class CriteriaArea : public Criteria<T>
{
public:
	CriteriaArea(double area);

	bool check(const std::shared_ptr<T>& item) const override;

private:
	double m_area;
};

#include "criteria_area_impl.cpp"

#endif
