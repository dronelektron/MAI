#include "trapezoid.h"

Trapezoid::Trapezoid()
{
	m_sideA = 0.0;
	m_sideB = 0.0;
	m_height = 0.0;
}

Trapezoid::Trapezoid(std::istream& is)
{
	std::cout << "================" << std::endl;
	std::cout << "Enter side A: ";
	is >> m_sideA;
	std::cout << "Enter side B: ";
	is >> m_sideB;
	std::cout << "Enter height: ";
	is >> m_height;
}

void Trapezoid::print() const
{
	std::cout << "================" << std::endl;
	std::cout << "Figure type: trapezoid" << std::endl;
	std::cout << "Side A size: " << m_sideA << std::endl;
	std::cout << "Side B size: " << m_sideB << std::endl;
	std::cout << "Height: " << m_height << std::endl;
}

double Trapezoid::area() const
{
	return m_height * (m_sideA + m_sideB) / 2.0;
}
