#include "trapezoid.h"

Trapezoid::Trapezoid()
{
	m_sideA = 0.0;
	m_sideB = 0.0;
	m_height = 0.0;
}

Trapezoid::Trapezoid(std::istream& is)
{
	is >> *this;
}

void Trapezoid::print() const
{
	std::cout << *this;
}

double Trapezoid::area() const
{
	return m_height * (m_sideA + m_sideB) / 2.0;
}

Trapezoid& Trapezoid::operator = (const Trapezoid& other)
{
	if (&other == this)
		return *this;

	m_sideA = other.m_sideA;
	m_sideB = other.m_sideB;
	m_height = other.m_height;

	return *this;
}

bool Trapezoid::operator == (const Trapezoid& other) const
{
	return m_sideA == other.m_sideA && m_sideB == other.m_sideB && m_height == other.m_height;
}

std::ostream& operator << (std::ostream& os, const Trapezoid& trapezoid)
{
	os << "================" << std::endl;
	os << "Figure type: trapezoid" << std::endl;
	os << "Side A size: " << trapezoid.m_sideA << std::endl;
	os << "Side B size: " << trapezoid.m_sideB << std::endl;
	os << "Height: " << trapezoid.m_height << std::endl;

	return os;
}

std::istream& operator >> (std::istream& is, Trapezoid& trapezoid)
{
	std::cout << "================" << std::endl;
	std::cout << "Enter side A: ";
	is >> trapezoid.m_sideA;
	std::cout << "Enter side B: ";
	is >> trapezoid.m_sideB;
	std::cout << "Enter height: ";
	is >> trapezoid.m_height;

	return is;
}
