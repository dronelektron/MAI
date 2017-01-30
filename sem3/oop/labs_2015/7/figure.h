#ifndef FIGURE_H
#define FIGURE_H

#include <cstring>
#include "allocator.h"

class Figure
{
public:
	virtual ~Figure();
	virtual void print() const = 0;
	virtual double area() const = 0;
	virtual const char* getName() const = 0;

	static Allocator allocator;
};

#endif
