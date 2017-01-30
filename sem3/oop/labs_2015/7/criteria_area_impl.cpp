template <class T>
CriteriaArea<T>::CriteriaArea(double area)
{
	m_area = area;
}

template <class T>
bool CriteriaArea<T>::check(const std::shared_ptr<T>& item) const
{
	return item->area() < m_area;
}
