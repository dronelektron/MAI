#include "container.h"
#include "square.h"
#include "rectangle.h"
#include "trapezoid.h"
#include "criteria_type.h"
#include "criteria_area.h"

int main()
{
	unsigned int action;
	Container<Figure> cont;

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
								cont.add(std::shared_ptr<Square>(new Square(std::cin)));

								break;
							}
							
							case 2:
							{
								cont.add(std::shared_ptr<Rectangle>(new Rectangle(std::cin)));

								break;
							}

							case 3:
							{
								cont.add(std::shared_ptr<Trapezoid>(new Trapezoid(std::cin)));

								break;
							}
						}
					}
				}

				break;
			}

			case 2:
			{
				unsigned int byCriteria;

				std::cout << "================" << std::endl;
				std::cout << "1) By type" << std::endl;
				std::cout << "2) By area" << std::endl;
				std::cout << "0) Quit" << std::endl;
				std::cin >> byCriteria;

				if (byCriteria > 0)
				{
					if (byCriteria > 2)
					{
						std::cout << "Error: invalid criteria" << std::endl;

						continue;
					}

					switch (byCriteria)
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
										cont.erase(CriteriaType<Figure>("Square"));

										break;
									}
									
									case 2:
									{
										cont.erase(CriteriaType<Figure>("Rectangle"));

										break;
									}

									case 3:
									{
										cont.erase(CriteriaType<Figure>("Trapezoid"));

										break;
									}
								}
							}
							
							break;
						}
						
						case 2:
						{
							double area;

							std::cout << "Enter area: ";
							std::cin >> area;

							cont.erase(CriteriaArea<Figure>(area));

							break;
						}
					}
				}

				break;
			}

			case 3:
			{
				std::cout << cont;

				break;
			}
		}
	}
	
	return 0;
}
