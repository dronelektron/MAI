#include "queue.h"

Queue::Queue()
{
	m_size = 0;
}

Queue::~Queue()
{
	while (size() > 0)
		pop();
}

void Queue::push(const std::shared_ptr<Figure>& figure)
{
	std::shared_ptr<QueueItem> itemPtr = std::make_shared<QueueItem>(figure);

	if (m_size == 0)
	{
		m_front = itemPtr;
		m_end = m_front;
	}
	else
	{
		m_end->setNext(itemPtr);
		m_end = itemPtr;
	}

	++m_size;
}

void Queue::pop()
{
	if (m_size == 1)
	{
		m_front = nullptr;
		m_end = nullptr;
	}
	else
		m_front = m_front->getNext();

	--m_size;
}

unsigned int Queue::size() const
{
	return m_size;
}

std::shared_ptr<Figure> Queue::front() const
{
	return m_front->getFigure();
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
		std::shared_ptr<QueueItem> item = queue.m_front;

		while (item != nullptr)
		{
			item->getFigure()->print();
			item = item->getNext();
		}
	}

	return os;
}
