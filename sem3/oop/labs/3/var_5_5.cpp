#include <iostream>
#include <string>
#include <thread>
#include <random>
#include <chrono>
#include <mutex>
#include "../../data_structures/vector.h"
#include "../../data_structures/queue.h"
#include "factory.h"

enum kFigure
{
	ROMB = 1,
	SIDE5,
	SIDE6
};

typedef std::pair<size_t, size_t> TimeArg;
typedef std::pair<std::string, size_t> Item;

struct Figure
{
	Item key;
	Shape* shape;
};

typedef ds::Queue<Figure> ContType2;
typedef ds::Vector<ContType2> ContType1;

void printDelimiter(char ch, size_t n);

void addFigure(ContType2& cont, const Figure& figure);
void deleteFigure(ContType2& cont, const ContType2::iterator& it);

void pushFigure(std::mutex& mtx, ContType1& cont, Factory& factory, kFigure figType);

void tFig(std::mutex& mtx, ContType1& cont, TimeArg timeData, const std::string& tName, kFigure fig, Factory& factory);
void tRemover(std::mutex& mtx, ContType1& cont, TimeArg timeData, double square, const std::string& tName);
void tPrinter(std::mutex& mtx, ContType1& cont, TimeArg timeData, const std::string& tName);

int main()
{
	double minSquare;
	TimeArg timeData1;
	TimeArg timeData2;
	TimeArg timeData3;
	TimeArg timeData4;
	TimeArg timeData5;
	ContType1 arr;
	Factory factory;
	
	std::cout << "Thread 1 (Romb) work time (sec): ";
	std::cin >> timeData1.first;
	std::cout << "Thread 1 (Romb) sleep time (msec): ";
	std::cin >> timeData1.second;
	
	std::cout << "Thread 2 (Side5) work time (sec): ";
	std::cin >> timeData2.first;
	std::cout << "Thread 2 (Side5) sleep time (msec): ";
	std::cin >> timeData2.second;

	std::cout << "Thread 3 (Side6) work time (sec): ";
	std::cin >> timeData3.first;
	std::cout << "Thread 3 (Side6) sleep time (msec): ";
	std::cin >> timeData3.second;

	std::cout << "Thread 4 (Remover) work time (sec): ";
	std::cin >> timeData4.first;
	std::cout << "Thread 4 (Remover) sleep time (msec): ";
	std::cin >> timeData4.second;
	std::cout << "Thread 4 (Remover) square: ";
	std::cin >> minSquare;

	std::cout << "Thread 5 (Printer) work time (sec): ";
	std::cin >> timeData5.first;
	std::cout << "Thread 5 (Printer) interval (sec): ";
	std::cin >> timeData5.second;
	std::cin.ignore();
	
	std::mutex mtx;

	std::thread t1(tFig, std::ref(mtx), std::ref(arr), timeData1, "Thread 1 (Romb)", ROMB, std::ref(factory));
	std::thread t2(tFig, std::ref(mtx), std::ref(arr), timeData2, "Thread 2 (Side5)", SIDE5, std::ref(factory));
	std::thread t3(tFig, std::ref(mtx), std::ref(arr), timeData3, "Thread 3 (Side6)", SIDE6, std::ref(factory));
	std::thread t4(tRemover, std::ref(mtx), std::ref(arr), timeData4, minSquare, "Thread 4 (Remover)");
	std::thread t5(tPrinter, std::ref(mtx), std::ref(arr), timeData5, "Thread 5 (Printer)");

	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	
	return 0;
}

void printDelimiter(char ch, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		std::cout << ch;

	std::cout << std::endl;
}

void addFigure(ContType2& cont, const Figure& figure)
{
	ContType2 q;

	while (!cont.empty() && cont.front().key <= figure.key)
	{
		q.push(cont.front());
		cont.pop();
	}

	q.push(figure);

	while (!cont.empty())
	{
		q.push(cont.front());
		cont.pop();
	}
	
	std::swap<ContType2>(cont, q);
}

void deleteFigure(ContType2& cont, const ContType2::iterator& it)
{
	ContType2 q;

	while (!cont.empty() && cont.begin() != it)
	{
		q.push(cont.front());
		cont.pop();
	}
	
	if (!cont.empty())
		cont.pop();

	while (!cont.empty())
	{
		q.push(cont.front());
		cont.pop();	
	}

	std::swap<ContType2>(cont, q);
}

void pushFigure(std::mutex& mtx, ContType1& cont, Factory& factory, kFigure figType)
{
	std::lock_guard<std::mutex> lockGuard(mtx);

	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::default_random_engine rndEng(seed);
	std::uniform_int_distribution<int> distr1(1, 32);
	std::uniform_int_distribution<int> distr2(32, 126);
	std::uniform_int_distribution<int> distr3(0, 2000000000);
	std::uniform_real_distribution<double> distr4(0.0, 50.0);

	Figure fig;

	size_t maxLen = distr1(rndEng);

	for (size_t i = 0; i < maxLen; ++i)
		fig.key.first += distr2(rndEng);

	fig.key.second = distr3(rndEng);
	fig.shape = factory.makeShape(figType);

	if (figType == ROMB)
	{
		double hor = distr4(rndEng);
		double ver = distr4(rndEng);

		Romb* romb = dynamic_cast<Romb*>(fig.shape);

		romb->setDiagHor(hor);
		romb->setDiagVer(ver);
	}
	else
	{
		double len = distr4(rndEng);

		Side5* sider = dynamic_cast<Side5*>(fig.shape);

		sider->setSide(len);
	}

	if (cont.empty() || cont[cont.size() - 1].size() == 5)
		cont.push_back(ContType2());

	addFigure(cont[cont.size() - 1], fig);

	printDelimiter('=', 64);
	std::cout << "ADDING:" << std::endl;
	printDelimiter('=', 64);
	std::cout << "Name: " << fig.key.first << std::endl;
	std::cout << "Id: " << fig.key.second << std::endl;

	fig.shape->printInfo();
}

void tFig(std::mutex& mtx, ContType1& cont, TimeArg timeData, const std::string& tName, kFigure fig, Factory& factory)
{
	std::unique_lock<std::mutex> ul(mtx);

	printDelimiter('=', 64);
	std::cout << tName << " started (" << timeData.first << ", " << timeData.second << ")" << std::endl;

	ul.unlock();

	timeData.first *= 1000;

	size_t start = 0;
	
	while (start < timeData.first)
	{
		pushFigure(mtx, cont, factory, fig);

		std::this_thread::sleep_for(std::chrono::milliseconds(timeData.second));

		start += timeData.second;
	}

	ul.lock();

	printDelimiter('=', 64);
	std::cout << tName << " finished" << std::endl;
}

void tRemover(std::mutex& mtx, ContType1& cont, TimeArg timeData, double square, const std::string& tName)
{
	std::unique_lock<std::mutex> ul(mtx);

	printDelimiter('=', 64);
	std::cout << tName << " started (" << timeData.first << ", " << timeData.second << ", " << square << ")" << std::endl;

	ul.unlock();
	
	Visitor visitor;

	timeData.first *= 1000;

	size_t start = 0;
	
	while (start < timeData.first)
	{
		ul.lock();
		
		for (size_t i = 0; i < cont.size(); ++i)
		{
			bool isFound = false;
			ContType2::iterator itDel;

			for (ContType2::iterator it = cont[i].begin(); it != cont[i].end(); ++it)
			{
				if (it->shape->accept(&visitor) < square)
				{
					printDelimiter('=', 64);
					std::cout << "REMOVING:" << std::endl;
					printDelimiter('=', 64);
					std::cout << "Name: " << it->key.first << std::endl;
					std::cout << "Id: " << it->key.second << std::endl;
					
					it->shape->printInfo();

					itDel = it;
					isFound = true;

					break;
				}
			}

			if (isFound)
			{
				deleteFigure(cont[i], itDel);

				if (cont[i].empty())
					cont.erase(i);

				i--;
			}
		}

		ul.unlock();

		std::this_thread::sleep_for(std::chrono::milliseconds(timeData.second));

		start += timeData.second;
	}
	
	ul.lock();

	printDelimiter('=', 64);
	std::cout << "Thread 4 (Remover) finished" << std::endl;
}

void tPrinter(std::mutex& mtx, ContType1& cont, TimeArg timeData, const std::string& tName)
{
	std::unique_lock<std::mutex> ul(mtx);

	printDelimiter('=', 64);
	std::cout << tName << " started (" << timeData.first << ", " << timeData.second << ")" << std::endl;

	ul.unlock();
	
	timeData.first *= 1000;

	size_t start = 0;
	
	while (start < timeData.first)
	{
		ul.lock();
		
		printDelimiter('=', 64);
		std::cout << "Printing" << std::endl;
		printDelimiter('=', 64);

		for (size_t i = 0; i < cont.size(); ++i)
		{
			std::cout << "----Container #" << (i + 1) << ":" << std::endl;

			for (ContType2::iterator it = cont[i].begin(); it != cont[i].end(); ++it)
			{
				std::cout << std::endl;
				std::cout << "Name: " << it->key.first << std::endl;
				std::cout << "Id: " << it->key.second << std::endl;

				it->shape->printInfo();
			}

			if (i + 1 < cont.size())
				std::cout << std::endl;
		}

		ul.unlock();

		std::this_thread::sleep_for(std::chrono::milliseconds(timeData.second));

		start += timeData.second;
	}
	
	ul.lock();

	printDelimiter('=', 64);
	std::cout << "Thread 5 (Printer) finished" << std::endl;
}
