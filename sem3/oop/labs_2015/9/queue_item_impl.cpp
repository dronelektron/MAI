template <class T>
QueueItem<T>::QueueItem(const std::shared_ptr<T>& item)
{
	m_item = item;
}

template <class T>
void QueueItem<T>::setNext(std::shared_ptr<QueueItem<T>> next)
{
	m_next = next;
}

template <class T>
std::shared_ptr<QueueItem<T>> QueueItem<T>::getNext()
{
	return m_next;
}

template <class T>
std::shared_ptr<T> QueueItem<T>::getItem() const
{
	return m_item;
}
