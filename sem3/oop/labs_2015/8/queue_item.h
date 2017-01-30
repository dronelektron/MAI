#ifndef QUEUE_ITEM_H
#define QUEUE_ITEM_H

#include <memory>

template <class T>
class QueueItem
{
public:
	QueueItem(const std::shared_ptr<T>& item);

	void setNext(std::shared_ptr<QueueItem<T>> next);
	std::shared_ptr<QueueItem<T>> getNext();
	std::shared_ptr<T> getItem() const;

private:
	std::shared_ptr<T> m_item;
	std::shared_ptr<QueueItem<T>> m_next;
};

#include "queue_item_impl.cpp"

#endif
