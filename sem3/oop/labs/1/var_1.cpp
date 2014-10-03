#include <iostream>
#include <string>
#include "../../data_structures/vector.h"

typedef std::pair<std::string, size_t> Item;
typedef ds::Vector<Item> Cont;

int main()
{
	size_t num;
	ds::Vector<Cont> v;

	while (true)
	{
		std::cout << "Menu" << std::endl;
		std::cout << "1) Add item" << std::endl;
		std::cout << "2) Find item" << std::endl;
		std::cout << "3) Remove item" << std::endl;
		std::cout << "4) Print all items" << std::endl;
		std::cout << "5) Exit" << std::endl;

		std::cout << "Choose action: ";
		std::cin >> num;
		std::cin.ignore();

		if (num == 5)
			break;

		if (num > 4)
		{
			std::cout << "Error. No such action in menu" << std::endl;

			continue;
		}

		switch (num)
		{
			case 1:
			{
				Item item;

				std::cout << "Name: ";
				std::getline(std::cin, item.first);
				std::cout << "Id: ";
				std::cin >> item.second;
				std::cin.ignore();
				
				if (v.empty() || v[v.size() - 1].size() == 5)
					v.push_back(Cont());

				Cont& cur = v[v.size() - 1];

				cur.push_back(item);

				for (size_t i = cur.size(); i > 1 && cur[i - 2].first > cur[i - 1].first; --i)
					std::swap<Item>(cur[i - 2], cur[i - 1]);

				break;
			}

			case 2:
			{
				bool isFound = false;
				Item item;

				std::cout << "Name: ";
				std::getline(std::cin, item.first);
				std::cout << "Id: ";
				std::cin >> item.second;
				std::cin.ignore();

				for (size_t i = 0; i < v.size(); ++i)
				{
					for (size_t j = 0; j < v[i].size(); ++j)
					{
						if (v[i][j] == item)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Name: " << v[i][j].first << std::endl;
							std::cout << "Id: " << v[i][j].second << std::endl;
							std::cout << "================================" << std::endl;

							isFound = true;
						}
					}
				}

				if (!isFound)
				{
					std::cout << "================================" << std::endl;
					std::cout << "Not found" << std::endl;
					std::cout << "================================" << std::endl;
				}

				break;
			}

			case 3:
			{
				bool isRemoved = false;
				Item item;

				std::cout << "Name: ";
				std::getline(std::cin, item.first);
				std::cout << "Id: ";
				std::cin >> item.second;
				std::cin.ignore();

				for (size_t i = 0; i < v.size(); ++i)
				{
					for (size_t j = 0; j < v[i].size(); ++j)
					{
						if (v[i][j] == item)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Object was deleted" << std::endl << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Name: " << v[i][j].first << std::endl;
							std::cout << "Id: " << v[i][j].second << std::endl;
							std::cout << "================================" << std::endl;

							v[i].erase(j);

							if (v[i].empty())
								v.erase(i);

							isRemoved = true;

							break;
						}
					}

					if (isRemoved)
						break;
				}
				
				break;
			}

			case 4:
			{
				std::cout << "================================" << std::endl;

				for (size_t i = 0; i < v.size(); ++i)
				{
					std::cout << "----Container #" << (i + 1) << ":" << std::endl;

					for (size_t j = 0; j < v[i].size(); ++j)
					{
						std::cout << std::endl;
						std::cout << "Name: " << v[i][j].first << std::endl;
						std::cout << "Id: " << v[i][j].second << std::endl;
					}

					if (i + 1 < v.size())
						std::cout << std::endl;
				}

				std::cout << "================================" << std::endl;

				break;
			}
		}
	}

	return 0;
}
