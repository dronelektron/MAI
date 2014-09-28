#include <iostream>
#include <string>
#include "../../data_structures/vector.h"
#include "../../data_structures/queue.h"

typedef std::pair<std::string, size_t> Item;
typedef ds::Queue<Item> Cont;
typedef ds::Queue<Item>::iterator ContIt;

void addItem(Cont& cont, const Item& item);
void deleteItem(Cont& cont, const Item& item);

int main()
{
	size_t action;
	ds::Vector<Cont> arr;

	while (true)
	{
		std::cout << "Menu" << std::endl;
		std::cout << "1) Add item" << std::endl;
		std::cout << "2) Find item" << std::endl;
		std::cout << "3) Remove item" << std::endl;
		std::cout << "4) Print all items" << std::endl;
		std::cout << "5) Exit" << std::endl;

		std::cout << "Choose action: ";
		std::cin >> action;
		std::cin.ignore();

		if (action == 5)
			break;

		if (action > 4)
		{
			std::cout << "Error. No such action in menu" << std::endl;

			continue;
		}

		switch (action)
		{
			case 1:
			{
				Item item;

				std::cout << "Name: ";
				std::getline(std::cin, item.first);
				std::cout << "Id: ";
				std::cin >> item.second;
				std::cin.ignore();
				
				if (arr.empty() || arr[arr.size() - 1].size() == 5)
					arr.push_back(Cont());

				addItem(arr[arr.size() - 1], item);

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

				for (size_t i = 0; i < arr.size(); ++i)
				{
					for (ContIt it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						if (*it == item)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Name: " << it->first << std::endl;
							std::cout << "Id: " << it->second << std::endl;
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

				for (size_t i = 0; i < arr.size(); ++i)
				{
					for (ContIt it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						if (*it == item)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Object was deleted" << std::endl << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Name: " << it->first << std::endl;
							std::cout << "Id: " << it->second << std::endl;
							std::cout << "================================" << std::endl;

							deleteItem(arr[i], item);

							if (arr[i].empty())
								arr.erase(i);

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

				for (size_t i = 0; i < arr.size(); ++i)
				{
					std::cout << "----Container #" << (i + 1) << ":" << std::endl;

					for (ContIt it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						std::cout << std::endl;
						std::cout << "Name: " << it->first << std::endl;
						std::cout << "Id: " << it->second << std::endl;
					}

					if (i + 1 < arr.size())
						std::cout << std::endl;
				}

				std::cout << "================================" << std::endl;

				break;
			}
		}
	}

	return 0;
}

void addItem(Cont& cont, const Item& item)
{
	Cont q;

	while (!cont.empty() && cont.front().first <= item.first)
	{
		q.push(cont.front());
		cont.pop();
	}

	q.push(item);

	while (!cont.empty())
	{
		q.push(cont.front());
		cont.pop();
	}
	
	std::swap<Cont>(cont, q);
}

void deleteItem(Cont& cont, const Item& item)
{
	if (cont.empty())
		return;

	size_t n = cont.size();
	size_t i = 0;

	while (i < n && cont.front() != item)
	{
		cont.push(cont.front());
		cont.pop();
		
		i++;
	}
	
	if (cont.front() == item)
		cont.pop();
}
