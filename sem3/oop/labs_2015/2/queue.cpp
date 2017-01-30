#include "queue.h"

Queue::Queue()
{
	m_front = nullptr;
	m_end = nullptr;
	m_size = 0;
}

Queue::~Queue()
{
	while (size() > 0)
		pop();
}

void Queue::push(const Square& square)
{
	QueueItem* item = new QueueItem(square);

	if (m_size == 0)
	{
		m_front = item;
		m_end = m_front;
	}
	else
	{
		m_end->setNext(item);
		m_end = item;
	}

	++m_size;
}

void Queue::pop()
{
	if (m_size == 1)
	{
		delete m_front;

		m_front = nullptr;
		m_end = nullptr;
	}
	else
	{
		QueueItem* next = m_front->getNext();

		delete m_front;

		m_front = next;
	}

	--m_size;
}

unsigned int Queue::size() const
{
	return m_size;
}

Square Queue::front() const
{
	return m_front->getSquare();
}

std::ostream& operator << (std::ostream& os, const Queue& queue)
{
	if (queue.size() == 0)
	{
		os << "================" << std::endl;
		os << "Queue is empty" << std::endl;
	}
	else
	{
		QueueItem* item = queue.m_front;

		while (item != nullptr)
		{
			os << item->getSquare();
			item = item->getNext();
		}
	}

	return os;
}
