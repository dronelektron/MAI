#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>
#include "queue_item.h"

class Queue
{
public:
	Queue();
	~Queue();

	void push(const std::shared_ptr<Figure>& figure);
	void pop();
	unsigned int size() const;
	std::shared_ptr<Figure> front() const;

	friend std::ostream& operator << (std::ostream& os, const Queue& queue);

private:
	std::shared_ptr<QueueItem> m_front;
	std::shared_ptr<QueueItem> m_end;
	unsigned int m_size;
};

#endif
