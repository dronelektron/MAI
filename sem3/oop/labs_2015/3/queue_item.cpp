#include "queue_item.h"

QueueItem::QueueItem(const std::shared_ptr<Figure>& figure)
{
	m_next = nullptr;
	m_figure = figure;
}

void QueueItem::setNext(std::shared_ptr<QueueItem> next)
{
	m_next = next;
}

std::shared_ptr<QueueItem> QueueItem::getNext()
{
	return m_next;
}

std::shared_ptr<Figure> QueueItem::getFigure() const
{
	return m_figure;
}
