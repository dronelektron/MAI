template <class T>
void Container<T>::add(const std::shared_ptr<T>& item)
{
	auto lastContIt = m_container.begin();
	
	if (lastContIt == m_container.end())
		m_container.push(std::make_shared<List<T>>());
	
	lastContIt = m_container.begin();

	while (lastContIt.getItem()->getNext() != nullptr)
		++lastContIt;

	if ((*lastContIt)->size() == 5)
	{
		m_container.push(std::make_shared<List<T>>());
		++lastContIt;
	}

	(*lastContIt)->add(item);

	for (unsigned int i = (*lastContIt)->size() - 1; i > 0; --i)
	{
		auto lastElemIt = (*lastContIt)->get(i);
		auto preLastElemIt = (*lastContIt)->get(i - 1);

		if (strcmp(preLastElemIt->getName(), lastElemIt->getName()) <= 0)
			break;

		preLastElemIt.getItem()->swap(*lastElemIt.getItem());
	}
}

template <class T>
void Container<T>::erase(const Criteria<T>& criteria)
{
	for (auto subCont : m_container)
	{
		while (true)
		{
			bool isRemoved = false;

			for (unsigned int i = 0; i < subCont->size(); ++i)
			{
				auto elemIt = subCont->get(i);

				if (criteria.check(*elemIt))
				{
					subCont->erase(elemIt);
					isRemoved = true;

					break;
				}
			}

			if (!isRemoved)
				break;
		}
	}

	while (m_container.size() > 0 && m_container.front()->size() == 0)
		m_container.pop();
	
	if (m_container.size() > 0)
	{
		auto firstSubCont = m_container.front();

		m_container.push(firstSubCont);
		m_container.pop();

		while (m_container.front() != firstSubCont)
		{
			if (m_container.front()->size() > 0)
				m_container.push(m_container.front());

			m_container.pop();
		}
	}
}

template <class K>
std::ostream& operator << (std::ostream& os, const Container<K>& container)
{
	if (container.m_container.size() == 0)
	{
		os << "================" << std::endl;
		os << "Container is empty" << std::endl;
	}
	else
	{
		unsigned int containerCnt1 = 1;

		for (auto subCont : container.m_container)
		{
			unsigned int containerCnt2 = 1;

			os << "================" << std::endl;
			os << "Container #" << (containerCnt1++) << ":" << std::endl;
			
			for (auto subItem : *subCont)
			{
				os << "================" << std::endl;
				os << "Item #" << (containerCnt2++) << ":" << std::endl;
				
				subItem->print();

				os << "Area: " << subItem->area() << std::endl;
			}
		}
	}

	return os;
}
