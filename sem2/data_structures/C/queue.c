#include "queue.h"

void queueCreate(Queue *q)
{
	q->_first = q->_last = (ItemQueue *)malloc(sizeof(ItemQueue));
	q->_size = 0;
}

int queueEmpty(const Queue *q)
{
	return q->_first == q->_last;
}

int queueSize(const Queue *q)
{
	return q->_size;
}

int queuePush(Queue *q, const int value)
{
	if (!(q->_last->_next = (ItemQueue *)malloc(sizeof(ItemQueue))))
		return 0;

	q->_last->_data = value;
	q->_last = q->_last->_next;
	q->_size++;

	return 1;
}

int queuePop(Queue *q)
{
	ItemQueue *item = NULL;

	if (queueEmpty(q))
		return 0;

	item = q->_first;
	q->_first = q->_first->_next;
	q->_size--;

	free(item);

	return 1;
}

int queueTop(const Queue *q)
{
	return q->_first->_data;
}

void queueDestroy(Queue *q)
{
	ItemQueue *item = q->_first;

	q->_first = q->_first->_next;
	q->_last->_next = NULL;

	free(item);

	if (q->_first)
		queueDestroy(q);

	q->_last = NULL;
	q->_size = 0;
}
