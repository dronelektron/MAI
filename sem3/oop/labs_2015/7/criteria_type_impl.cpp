template <class T>
CriteriaType<T>::CriteriaType(const char* type)
{
	strcpy(m_type, type);
}

template <class T>
bool CriteriaType<T>::check(const std::shared_ptr<T>& item) const
{
	return strcmp(m_type, item->getName()) == 0;
}
