#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <iostream>
#include "figure.h"

class Rectangle : public Figure
{
public:
	Rectangle();
	Rectangle(std::istream& is);

	void print() const override;
	double area() const override;

	Rectangle& operator = (const Rectangle& other);
	bool operator == (const Rectangle& other) const;

	friend std::ostream& operator << (std::ostream& os, const Rectangle& rectangle);
	friend std::istream& operator >> (std::istream& is, Rectangle& rectangle);

private:
	double m_sideA;
	double m_sideB;
};

#endif
