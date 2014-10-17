#ifndef FACTORY_H
#define FACTORY_H

#include "shape.h"

class Factory
{
public:
	Shape* makeShape(size_t index);
};

#endif
