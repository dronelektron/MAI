#ifndef ITERATOR_H
#define ITERATOR_H

template <class N, class T>
class Iterator
{
public:
	Iterator(const std::shared_ptr<N>& item);

	std::shared_ptr<N> getItem() const;

	std::shared_ptr<T> operator * ();
	std::shared_ptr<T> operator -> ();
	Iterator operator ++ ();
	Iterator operator ++ (int index);
	bool operator == (const Iterator& other) const;
	bool operator != (const Iterator& other) const;

private:
	std::shared_ptr<N> m_item;
};

#include "iterator_impl.cpp"

#endif
