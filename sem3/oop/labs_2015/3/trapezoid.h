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

	Trapezoid& operator = (const Trapezoid& other);
	bool operator == (const Trapezoid& other) const;

	friend std::ostream& operator << (std::ostream& os, const Trapezoid& trapezoid);
	friend std::istream& operator >> (std::istream& is, Trapezoid& trapezoid);

private:
	double m_sideA;
	double m_sideB;
	double m_height;
};

#endif
