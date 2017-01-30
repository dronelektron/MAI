#include "square.h"
#include "rectangle.h"
#include "trapezoid.h"

void testFigure(Figure* figure);

int main()
{
	testFigure(new Square(std::cin));
	testFigure(new Rectangle(std::cin));
	testFigure(new Trapezoid(std::cin));

	return 0;
}

void testFigure(Figure* figure)
{
	figure->print();

	std::cout << "Area: " << figure->area() << std::endl;

	delete figure;
}