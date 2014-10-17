#include "factory.h"

Shape* Factory::makeShape(size_t index)
{
	Shape* sh = NULL;

	switch (index)
	{
		case 1: sh = new Romb; break;
		case 2: sh = new Side5; break;
		case 3: sh = new Side6; break;
	}

	return sh;
}
