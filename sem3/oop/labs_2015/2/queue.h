#ifndef QUEUE_H
#define QUEUE_H

#include "queue_item.h"

class Queue
{
public:
	Queue();
	~Queue();

	void push(const Square& square);
	void pop();
	unsigned int size() const;
	Square front() const;

	friend std::ostream& operator << (std::ostream& os, const Queue& queue);

private:
	QueueItem* m_front;
	QueueItem* m_end;
	unsigned int m_size;
};

#endif
