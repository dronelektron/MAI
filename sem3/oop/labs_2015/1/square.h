#ifndef SQUARE_H
#define SQUARE_H

#include <iostream>
#include "figure.h"

class Square : public Figure
{
public:
	Square();
	Square(std::istream& is);

	void print() const override;
	double area() const override;

private:
	double m_side;
};

#endif
