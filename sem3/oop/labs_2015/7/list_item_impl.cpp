template <class T>
ListItem<T>::ListItem(const std::shared_ptr<T>& item)
{
	m_item = item;
}

template <class T>
void ListItem<T>::swap(ListItem<T>& other)
{
	m_item.swap(other.m_item);
}

template <class T>
void ListItem<T>::setPrev(std::shared_ptr<ListItem<T>> prev)
{
	m_prev = prev;
}

template <class T>
void ListItem<T>::setNext(std::shared_ptr<ListItem<T>> next)
{
	m_next = next;
}

template <class T>
std::shared_ptr<ListItem<T>> ListItem<T>::getPrev()
{
	return m_prev;
}

template <class T>
std::shared_ptr<ListItem<T>> ListItem<T>::getNext()
{
	return m_next;
}

template <class T>
std::shared_ptr<T> ListItem<T>::getItem() const
{
	return m_item;
}
