#include "cudaTest.h"
#include "CImg.h"
#include <iostream>

using namespace cimg_library;

int main()
{

	// create Cuda tester
	auto cudatest = new CudaTest();

	// Construct a 256x256x3 grayscale volume, filled with value '1'.
	CImg<float> volume(2, 2, 1, 1, 1); 
	int size[3] = { volume.width(), volume.height(), volume.depth() };
	float origin[3] = { .0f, .0f, .0f };
	cudatest->copyVolumeToGPU(size, origin, volume.data());
	
	// construct a projection geometry
	float src[3] = { -1.5f, .0f, .0f };
	auto detElements = 1;
	float det[3] = { 2.5, 3.5, 0 };
	float value = 0;
	auto arr = &value;
	printf("pts_main: %.2f,%.2f,%.2f\n", det[0], det[1], det[2]);
	cudatest->copyProjectorToGPU(src, detElements, &det[0], &det[1], &det[2], arr);

	delete cudatest;

}
