#include "image.h"

cudaArray* g_arr;
Texture2D g_tex;

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

void imageCreateTexture(Image* image)
{
	int w = image->width;
	int h = image->height;

	g_tex.channelDesc = cudaCreateChannelDesc<Pixel>();
	g_tex.addressMode[0] = cudaAddressModeClamp;
	g_tex.addressMode[1] = cudaAddressModeClamp;
	g_tex.filterMode = cudaFilterModePoint;
	g_tex.normalized = false;

	ERR(cudaMallocArray(&g_arr, &g_tex.channelDesc, w, h));
	ERR(cudaMemcpyToArray(g_arr, 0, 0, image->pixels, sizeof(Pixel) * w * h, cudaMemcpyHostToDevice));
	ERR(cudaBindTextureToArray(g_tex, g_arr, g_tex.channelDesc));
}

void imageDeleteTexture()
{
	ERR(cudaUnbindTexture(g_tex));
	ERR(cudaFreeArray(g_arr));
}
