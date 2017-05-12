#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include "err.h"

typedef const char* CString;
typedef unsigned char Byte;
typedef uchar4 Pixel;
typedef texture<Pixel, 2, cudaReadModeElementType> Texture2D;

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
void imageCreateTexture(Image* image);
void imageDeleteTexture();

extern cudaArray* g_arr;
extern Texture2D g_tex;

#endif
