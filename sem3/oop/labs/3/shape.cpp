#include "shape.h"

Shape::~Shape() {}

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

void Romb::randomize(std::default_random_engine& rnd, std::uniform_real_distribution<double>& urd)
{
	_diagHor = urd(rnd);
	_diagVer = urd(rnd);
}

double Romb::getDiagHor() const
{
	return _diagHor;
}

double Romb::getDiagVer() const
{
	return _diagVer;
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

void Side5::randomize(std::default_random_engine& rnd, std::uniform_real_distribution<double>& urd)
{
	_side = urd(rnd);
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
