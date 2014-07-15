#ifndef FUNCS_H
#define FUNCS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generateArray(int *a, const int n, int from, int to);
void shuffleArray(int *a, const int n);
void printArray(int *a, const int n);
void swap(int *a, int *b);
int randomAB(int a, int b);

#endif
