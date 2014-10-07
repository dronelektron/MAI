#include "factory.h"

Shape* Factory::makeShape(const std::string& shapeName)
{
	if (shapeName == "Romb")
		return new Romb;

	if (shapeName == "Side5")
		return new Side5;

	if (shapeName == "Side6")
		return new Side6;

	return NULL;
}
