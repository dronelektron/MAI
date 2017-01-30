#include "allocator.h"

Allocator::Allocator(unsigned int blockSize, unsigned int count)
{
	m_memory = malloc(blockSize * count);

	for (unsigned int i = 0; i < count; ++i)
		m_freeBlocks.add(std::make_shared<unsigned int>(i * blockSize));
}

Allocator::~Allocator()
{
	free(m_memory);
}

void* Allocator::allocate()
{
	void* res = R_CAST(R_CAST(m_memory, char*) + **m_freeBlocks.get(0), void*);

	m_freeBlocks.erase(m_freeBlocks.begin());

	return res;
}

void Allocator::deallocate(void* p)
{
	unsigned int offset = R_CAST(p, char*) - R_CAST(m_memory, char*);
	
	m_freeBlocks.add(std::make_shared<unsigned int>(offset));
}

bool Allocator::hasFreeBlocks() const
{
	return m_freeBlocks.size() > 0;
}
