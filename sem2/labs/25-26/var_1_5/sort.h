#ifndef UDT_SORT_H
#define UDT_SORT_H

#include "udt.h"

void udtMergeSort(Udt *udt);
void udtPartition(Udt *udt, Udt *udt1, Udt *udt2);
void udtMerge(Udt *udt, Udt *udt1, Udt *udt2);

#endif
