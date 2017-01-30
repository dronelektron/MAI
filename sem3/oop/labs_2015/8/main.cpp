#include "queue.h"
#include "square.h"
#include "rectangle.h"
#include "trapezoid.h"

int main()
{
	unsigned int action;
	Queue<Figure> q;

	while (true)
	{
		std::cout << "================" << std::endl;
		std::cout << "Menu:" << std::endl;
		std::cout << "1) Add figure" << std::endl;
		std::cout << "2) Delete figure" << std::endl;
		std::cout << "3) Print" << std::endl;
		std::cout << "4) Sort" << std::endl;
		std::cout << "0) Quit" << std::endl;
		std::cin >> action;

		if (action == 0)
			break;
		
		if (action > 4)
		{
			std::cout << "Error: invalid action" << std::endl;

			continue;
		}

		switch (action)
		{
			case 1:
			{
				if (!Figure::allocator.hasFreeBlocks())
					std::cout << "Error. No free blocks" << std::endl;
				else
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
								q.push(std::shared_ptr<Square>(new Square(std::cin)));

								break;
							}
							
							case 2:
							{
								q.push(std::shared_ptr<Rectangle>(new Rectangle(std::cin)));

								break;
							}

							case 3:
							{
								q.push(std::shared_ptr<Trapezoid>(new Trapezoid(std::cin)));

								break;
							}
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

			case 4:
			{
				unsigned int sortType;

				std::cout << "================" << std::endl;
				std::cout << "1) Single thread" << std::endl;
				std::cout << "2) Multithread" << std::endl;
				std::cout << "0) Quit" << std::endl;
				std::cin >> sortType;

				if (sortType > 0)
				{
					if (sortType > 2)
					{
						std::cout << "Error: invalid sort type" << std::endl;

						continue;
					}

					switch (sortType)
					{
						case 1:
						{
							q.sort();

							break;
						}
						
						case 2:
						{
							q.sortParallel();

							break;
						}
					}
				}
				
				break;
			}
		}
	}

	return 0;
}
