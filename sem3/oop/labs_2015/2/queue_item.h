#ifndef QUEUE_ITEM_H
#define QUEUE_ITEM_H

#include "square.h"

class QueueItem
{
public:
	QueueItem(const Square& square);

	void setNext(QueueItem* next);
	QueueItem* getNext();
	Square getSquare() const;

private:
	Square m_square;
	QueueItem* m_next;
};

#endif
