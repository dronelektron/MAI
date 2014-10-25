#include "shape.h"

Shape::~Shape() {}

Romb::Romb()
{
	_diagHor = 0.0;
	_diagVer = 0.0;
}

double Romb::accept(Visitor* vis)
{
	return vis->visit(this);
}

void Romb::printInfo()
{
	std::cout << "Figure: Romb" << std::endl;
	std::cout << "Len of hor diagonal: " << _diagHor << std::endl;
	std::cout << "Len of ver diagonal: " << _diagVer << std::endl;
}

void Romb::setDiagHor(double val)
{
	_diagHor = val;
}

void Romb::setDiagVer(double val)
{
	_diagVer = val;
}

double Romb::getDiagHor() const
{
	return _diagHor;
}

double Romb::getDiagVer() const
{
	return _diagVer;
}

Side5::Side5()
{
	_side = 0.0;
}

double Side5::accept(Visitor* vis)
{
	return vis->visit(this);
}

void Side5::printInfo()
{
	std::cout << "Figure: Side5" << std::endl;
	std::cout << "Len of side: " << _side << std::endl;
}

void Side5::setSide(double side)
{
	_side = side;
}

double Side5::getSide() const
{
	return _side;
}

double Side6::accept(Visitor* vis)
{
	return vis->visit(this);
}

void Side6::printInfo()
{
	std::cout << "Figure: Side6" << std::endl;
	std::cout << "Len of side: " << _side << std::endl;
}
