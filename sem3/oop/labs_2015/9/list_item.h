#ifndef LIST_ITEM_H
#define LIST_ITEM_H

#include <memory>

template <class T>
class ListItem
{
public:
	ListItem(const std::shared_ptr<T>& item);

	void setPrev(std::shared_ptr<ListItem<T>> prev);
	void setNext(std::shared_ptr<ListItem<T>> next);
	std::shared_ptr<ListItem<T>> getPrev();
	std::shared_ptr<ListItem<T>> getNext();
	std::shared_ptr<T> getItem() const;

private:
	std::shared_ptr<T> m_item;
	std::shared_ptr<ListItem<T>> m_prev;
	std::shared_ptr<ListItem<T>> m_next;
};

#include "list_item_impl.cpp"

#endif
