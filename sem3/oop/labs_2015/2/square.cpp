#include "square.h"

Square::Square()
{
	m_side = 0.0;
}

Square::Square(std::istream& is)
{
	is >> *this;
}

void Square::print() const
{
	std::cout << *this;
}

double Square::area() const
{
	return m_side * m_side;
}

Square& Square::operator = (const Square& other)
{
	if (&other == this)
		return *this;

	m_side = other.m_side;

	return *this;
}

bool Square::operator == (const Square& other) const
{
	return m_side == other.m_side;
}

std::ostream& operator << (std::ostream& os, const Square& square)
{
	os << "================" << std::endl;
	os << "Figure type: square" << std::endl;
	os << "Side size: " << square.m_side << std::endl;

	return os;
}

std::istream& operator >> (std::istream& is, Square& square)
{
	std::cout << "================" << std::endl;
	std::cout << "Enter side: ";
	is >> square.m_side;

	return is;
}
