#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

typedef struct _Image
{
	int width;
	int height;
	Pixel* pixels;
} Image;

void imageCreate(Image* image, int width, int height);
void imageDelete(Image* image);
void imageReadFromFile(Image* image, CString fileName);
void imageWriteToFile(Image* image, CString fileName);
Pixel imageGetPixel(Image* image, int row, int col);
int imageSize(Image* image);

#endif
