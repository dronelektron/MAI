#include "queue_item.h"

QueueItem::QueueItem(const Square& square)
{
	m_next = nullptr;
	m_square = square;
}

void QueueItem::setNext(QueueItem* next)
{
	m_next = next;
}

QueueItem* QueueItem::getNext()
{
	return m_next;
}

Square QueueItem::getSquare() const
{
	return m_square;
}
