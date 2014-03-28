#ifndef QUEUE_H
#define QUEUE_H

#include <stdlib.h>

typedef struct _ItemQueue
{
	int _data;
	struct _ItemQueue *_next;
} ItemQueue;

typedef struct _Queue
{
	struct _ItemQueue *_first;
	struct _ItemQueue *_last;
	int _size;
} Queue;

void queueCreate(Queue *q);
int queueEmpty(const Queue *q);
int queueSize(const Queue *q);
int queuePush(Queue *q, const int value);
int queuePop(Queue *q);
int queueTop(const Queue *q);
void queueDestroy(Queue *q);

#endif
