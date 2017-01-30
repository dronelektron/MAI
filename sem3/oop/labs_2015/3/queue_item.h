#ifndef QUEUE_ITEM_H
#define QUEUE_ITEM_H

#include <memory>
#include "figure.h"

class QueueItem
{
public:
	QueueItem(const std::shared_ptr<Figure>& figure);

	void setNext(std::shared_ptr<QueueItem> next);
	std::shared_ptr<QueueItem> getNext();
	std::shared_ptr<Figure> getFigure() const;

private:
	std::shared_ptr<Figure> m_figure;
	std::shared_ptr<QueueItem> m_next;
};

#endif
