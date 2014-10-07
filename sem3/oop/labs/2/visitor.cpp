#include "visitor.h"
#include "shape.h"

double Visitor::visit(Romb* shape)
{
	double s = shape->getDiagHor() * shape->getDiagVer() / 2.0;

	return s;
}

double Visitor::visit(Side5* shape)
{
	const double n = 5;
	double a = shape->getSide();
	double s = n * pow(a, 2.0) / (4.0 * tan(M_PI / n));

	return s;
}

double Visitor::visit(Side6* shape)
{
	const double n = 6;
	double a = shape->getSide();
	double s = n * pow(a, 2.0) / (4.0 * tan(M_PI / n));

	return s;
}
