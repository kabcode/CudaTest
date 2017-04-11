#ifndef CUDATEST_H
#define CUDATEST_H

#include <chrono>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <exception>
#include <string>

// typedef for readability
typedef float VoxelType;
const int DIM = 3;

// typedef for time checking
typedef std::chrono::high_resolution_clock hr_clock;

enum DEVICE
{
	NOTAVAILABLE,
	AVAILABLE,
	DATALOADED,
	DATAANDGEOMETRYLOADED
};

class CudaTest
{
public:
	// constructor
	explicit CudaTest();

	// destructor
	~CudaTest();

	// functions
	static void showDevices(int);
	int getGPUState() const { return dev_State; }
	void copyVolumeToGPU (const int*, const float*, const VoxelType*);
	void copyProjectorToGPU(const float*, const int, const float*, const float*, const float*, float*);

private:
	// attributes
	// GPU identificator
	int dev_ID;
	int dev_State;
	// pointer to memory at the device
	cudaArray* dev_Array;
	int host_dimension[DIM];

	// timing function
	void timing(std::chrono::duration<long long, std::nano>, std::string) const;

};

#endif //!CUDATEST_H


