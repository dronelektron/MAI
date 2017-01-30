#ifndef CRITERIA_H
#define CRITERIA_H

template <class T>
class Criteria
{
public:
	virtual bool check(const std::shared_ptr<T>& item) const = 0;
};

#endif
