#include "rectangle.h"

Rectangle::Rectangle()
{
	m_sideA = 0.0;
	m_sideB = 0.0;
}

Rectangle::Rectangle(std::istream& is)
{
	std::cout << "================" << std::endl;
	std::cout << "Enter side A: ";
	is >> m_sideA;
	std::cout << "Enter side B: ";
	is >> m_sideB;
}

void Rectangle::print() const
{
	std::cout << "================" << std::endl;
	std::cout << "Figure type: rectangle" << std::endl;
	std::cout << "Side A size: " << m_sideA << std::endl;
	std::cout << "Side B size: " << m_sideB << std::endl;
}

double Rectangle::area() const
{
	return m_sideA * m_sideB;
}
