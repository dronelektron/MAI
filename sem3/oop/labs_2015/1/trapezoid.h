#ifndef TRAPEZOID_H
#define TRAPEZOID_H

#include <iostream>
#include "figure.h"

class Trapezoid : public Figure
{
public:
	Trapezoid();
	Trapezoid(std::istream& is);

	void print() const override;
	double area() const override;

private:
	double m_sideA;
	double m_sideB;
	double m_height;
};

#endif
