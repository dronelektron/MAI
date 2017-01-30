#include "queue.h"
#include "square.h"
#include "rectangle.h"
#include "trapezoid.h"

int main()
{
	unsigned int action;
	Queue q;

	while (true)
	{
		std::cout << "================" << std::endl;
		std::cout << "Menu:" << std::endl;
		std::cout << "1) Add figure" << std::endl;
		std::cout << "2) Delete figure" << std::endl;
		std::cout << "3) Print" << std::endl;
		std::cout << "0) Quit" << std::endl;
		std::cin >> action;

		if (action == 0)
			break;
		
		if (action > 3)
		{
			std::cout << "Error: invalid action" << std::endl;

			continue;
		}

		switch (action)
		{
			case 1:
			{
				unsigned int figureType;

				std::cout << "================" << std::endl;
				std::cout << "1) Square" << std::endl;
				std::cout << "2) Rectangle" << std::endl;
				std::cout << "3) Trapezoid" << std::endl;
				std::cout << "0) Quit" << std::endl;
				std::cin >> figureType;

				if (figureType > 0)
				{
					if (figureType > 3)
					{
						std::cout << "Error: invalid figure type" << std::endl;

						continue;
					}

					switch (figureType)
					{
						case 1:
						{
							q.push(std::make_shared<Square>(std::cin));

							break;
						}
						
						case 2:
						{
							q.push(std::make_shared<Rectangle>(std::cin));

							break;
						}

						case 3:
						{
							q.push(std::make_shared<Trapezoid>(std::cin));

							break;
						}
					}
				}

				break;
			}

			case 2:
			{
				q.pop();

				break;
			}

			case 3:
			{
				std::cout << q;

				break;
			}
		}
	}
	
	return 0;
}
