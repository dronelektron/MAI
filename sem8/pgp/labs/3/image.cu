#include "image.h"

void imageCreate(Image* image, int width, int height)
{
	image->width = width;
	image->height = height;
	image->pixels = (Pixel*)malloc(sizeof(Pixel) * width * height);
}

void imageDelete(Image* image)
{
	free(image->pixels);

	image->width = 0;
	image->height = 0;
	image->pixels = NULL;
}

void imageReadFromFile(Image* image, CString fileName)
{
	FILE* file = fopen(fileName, "rb");
	int width;
	int height;

	fread(&width, sizeof(width), 1, file);
	fread(&height, sizeof(height), 1, file);
	imageCreate(image, width, height);
	fread(image->pixels, sizeof(Pixel), image->width * image->height, file);
	fclose(file);
}

void imageWriteToFile(Image* image, CString fileName)
{
	FILE* file = fopen(fileName, "wb");

	fwrite(&image->width, sizeof(image->width), 1, file);
	fwrite(&image->height, sizeof(image->height), 1, file);
	fwrite(image->pixels, sizeof(Pixel), image->width * image->height, file);
	fclose(file);
}

Pixel imageGetPixel(Image* image, int row, int col)
{
	return image->pixels[row * image->width + col];
}

int imageSize(Image* image)
{
	return sizeof(Pixel) * image->width * image->height;
}
