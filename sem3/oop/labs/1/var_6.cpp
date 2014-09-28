#include <iostream>
#include <string>
#include "../../data_structures/vector.h"
#include "../../data_structures/stack.h"

typedef std::pair<std::string, size_t> Item;
typedef ds::Stack<Item> Cont;

int main()
{
	size_t action;
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
				
				if (v.empty() || v[v.size() - 1].size() == 5)
					v.push_back(Cont());

				Cont st;
				Cont& lastCont = v[v.size() - 1];
	
				while (!lastCont.empty() && lastCont.top().first > item.first)
				{
					st.push(lastCont.top());
					lastCont.pop();
				}
	
				lastCont.push(item);

				while (!st.empty())
				{
					lastCont.push(st.top());
					st.pop();
				}
				
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
					Cont st;

					while (!v[i].empty() && v[i].top() != item)
					{
						st.push(v[i].top());
						v[i].pop();
					}

					if (!v[i].empty())
					{
						Item& cur = v[i].top();

						std::cout << "================================" << std::endl;
						std::cout << "Container #" << (i + 1) << ":" << std::endl;
						std::cout << "Name: " << cur.first << std::endl;
						std::cout << "Id: " << cur.second << std::endl;
						std::cout << "================================" << std::endl;
						
						isFound = true;
					}

					while (!st.empty())
					{
						v[i].push(st.top());
						st.pop();
					}

					if (isFound)
						break;
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
					Cont st;
					
					while (!v[i].empty() && v[i].top() != item)
					{
						st.push(v[i].top());
						v[i].pop();
					}

					if (!v[i].empty())
					{
						Item& cur = v[i].top();

						std::cout << "================================" << std::endl;
						std::cout << "Object was deleted" << std::endl << std::endl;
						std::cout << "Container #" << (i + 1) << ":" << std::endl;
						std::cout << "Name: " << cur.first << std::endl;
						std::cout << "Id: " << cur.second << std::endl;
						std::cout << "================================" << std::endl;

						v[i].pop();

						isRemoved = true;
					}
					
					while (!st.empty())
					{
						v[i].push(st.top());
						st.pop();
					}

					if (v[i].empty())
						v.erase(i);

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

					Cont st;

					while (!v[i].empty())
					{
						st.push(v[i].top());
						v[i].pop();
					}

					while (!st.empty())
					{
						Item& cur = st.top();
						
						std::cout << std::endl;
						std::cout << "Name: " << cur.first << std::endl;
						std::cout << "Id: " << cur.second << std::endl;
						
						v[i].push(cur);
						st.pop();
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
