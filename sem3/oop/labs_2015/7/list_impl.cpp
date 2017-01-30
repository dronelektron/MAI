template <class T>
List<T>::List()
{
	m_size = 0;
}

template <class T>
List<T>::~List()
{
	while (size() > 0)
		erase(begin());
}

template <class T>
void List<T>::add(const std::shared_ptr<T>& item)
{
	std::shared_ptr<ListItem<T>> itemPtr = std::make_shared<ListItem<T>>(item);

	if (m_size == 0)
	{
		m_begin = itemPtr;
		m_end = m_begin;
	}
	else
	{
		itemPtr->setPrev(m_end);
		m_end->setNext(itemPtr);
		m_end = itemPtr;
	}

	++m_size;
}

template <class T>
void List<T>::erase(const Iterator<ListItem<T>, T>& it)
{
	if (m_size == 1)
	{
		m_begin = nullptr;
		m_end = nullptr;
	}
	else
	{
		std::shared_ptr<ListItem<T>> left = it.getItem()->getPrev();
		std::shared_ptr<ListItem<T>> right = it.getItem()->getNext();
		std::shared_ptr<ListItem<T>> mid = it.getItem();

		mid->setPrev(nullptr);
		mid->setNext(nullptr);

		if (left != nullptr)
			left->setNext(right);
		else
			m_begin = right;

		if (right != nullptr)
			right->setPrev(left);
		else
			m_end = left;
	}

	--m_size;
}

template <class T>
unsigned int List<T>::size() const
{
	return m_size;
}

template <class T>
Iterator<ListItem<T>, T> List<T>::get(unsigned int index) const
{
	if (index >= size())
		return end();

	Iterator<ListItem<T>, T> it = begin();

	while (index > 0)
	{
		++it;
		--index;
	}

	return it;
}

template <class T>
Iterator<ListItem<T>, T> List<T>::begin() const
{
	return Iterator<ListItem<T>, T>(m_begin);
}

template <class T>
Iterator<ListItem<T>, T> List<T>::end() const
{
	return Iterator<ListItem<T>, T>(nullptr);
}

template <class K>
std::ostream& operator << (std::ostream& os, const List<K>& list)
{
	if (list.size() == 0)
	{
		os << "================" << std::endl;
		os << "List is empty" << std::endl;
	}
	else
		for (std::shared_ptr<K> item : list)
			item->print();

	return os;
}
