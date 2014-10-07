#ifndef FACTORY_H
#define FACTORY_H

#include <string>
#include "shape.h"

class Factory
{
public:
	Shape* makeShape(const std::string& shapeName);
};

#endif
