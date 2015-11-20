/*
Пример применения паттерна проектирования Singleton в проекте.

Можно использовать Singleton для базы данных, чтобы везде было доступно только одно активное соединение
*/

#include <iostream>
#include <string>

class Connection
{
private:
	Connection();
	Connection(const Connection& con);
	Connection& operator = (const Connection& con);

public:
	static Connection& getInstance();

	void action();

private:
	std::string m_host;
	int m_port;
};

int main()
{
	Connection& con = Connection::getInstance();

	con.action();

	return 0;
}

Connection::Connection()
{
	m_host = "localhost";
	m_port = 1234;
}

Connection& Connection::getInstance()
{
	static Connection con;

	return con;
}

void Connection::action()
{
	std::cout << "Connection action" << std::endl;
}
