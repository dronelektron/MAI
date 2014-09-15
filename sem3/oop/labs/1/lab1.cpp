#include <iostream>
#include <string>
#include "../../data_structures/vector.h"
#include "../../data_structures/queue.h"

typedef std::pair<size_t, std::string> Item;
typedef ds::Queue<Item> Cont;
typedef ds::Queue<Item>::iterator ContIter;

void sortMerge(ds::Queue<Item>& q);
void deleteItem(ds::Queue<Item>& q, std::string& name);

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
				size_t id;
				std::string name;

				std::cout << "Id: ";
				std::cin >> id;
				std::cin.ignore();
				std::cout << "Name: ";
				std::getline(std::cin, name);
				
				if (arr.empty() || arr[arr.size() - 1].size() % 5 == 0)
					arr.push_back(Cont());

				arr[arr.size() - 1].push(Item(id, name));

				sortMerge(arr[arr.size() - 1]);

				break;
			}

			case 2:
			{
				bool isFound = false;
				std::string name;

				std::cout << "Name: ";
				std::getline(std::cin, name);

				for (size_t i = 0; i < arr.size(); ++i)
				{
					for (ContIter it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						if (it->second == name)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Id: " << it->first << std::endl;
							std::cout << "Name: " << it->second << std::endl;
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
				std::string name;

				std::cout << "Name: ";
				std::getline(std::cin, name);

				for (size_t i = 0; i < arr.size(); ++i)
				{
					for (ContIter it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						if (it->second == name)
						{
							std::cout << "================================" << std::endl;
							std::cout << "Object was deleted" << std::endl << std::endl;
							std::cout << "Container #" << (i + 1) << ":" << std::endl;
							std::cout << "Id: " << it->first << std::endl;
							std::cout << "Name: " << it->second << std::endl;
							std::cout << "================================" << std::endl;

							deleteItem(arr[i], name);

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

					for (ContIter it = arr[i].begin(); it != arr[i].end(); ++it)
					{
						std::cout << std::endl;
						std::cout << "Id: " << it->first << std::endl;
						std::cout << "Name: " << it->second << std::endl;
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

void sortMerge(ds::Queue<Item>& q)
{
	if (q.size() < 2)
		return;

	size_t n = q.size();
	ds::Queue<Item> left;
	ds::Queue<Item> right;

	for (size_t i = 0; i < n / 2; ++i)
	{
		left.push(q.front());
		q.pop();
	}

	for (size_t i = n / 2; i < n; ++i)
	{
		right.push(q.front());
		q.pop();
	}

	sortMerge(left);
	sortMerge(right);

	while (!left.empty() && !right.empty())
	{
		if (left.front().second <= right.front().second)
		{
			q.push(left.front());
			left.pop();
		}
		else
		{
			q.push(right.front());
			right.pop();
		}
	}

	while (!left.empty())
	{
		q.push(left.front());
		left.pop();
	}

	while (!right.empty())
	{
		q.push(right.front());
		right.pop();
	}
}

void deleteItem(ds::Queue<Item>& q, std::string& name)
{
	if (q.empty())
		return;

	size_t n = q.size();
	size_t i = 0;

	while (i < n && q.front().second != name)
	{
		q.push(q.front());
		q.pop();
		
		i++;
	}
	
	if (q.front().second == name)
		q.pop();
}
