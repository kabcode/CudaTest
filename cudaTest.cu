#include "cudaTest.h"
#include <string>

// correction variables
#define RAY_TOLERANCE 1e-15f;
#define STRIDE_TOLERANCE 1e-15f
#define PLANE_IDX_TOLERANCE 1e-30f

// texture memory on the device, read only in a single kernel
texture<float, cudaTextureType3D, cudaReadModeElementType> dev_tex;
// device constants which will be connected to host constants
// the __constant__ variables are put in constant memory for fast access
__constant__ int dev_dimension[DIM];     // dimension of volume on device
__constant__ float dev_volumeOrigin[DIM];// offset of the volume
__constant__ float dev_source[DIM];      // source position
__constant__ float dev_volPlaneMin[DIM]; // bounding box planes minimum
__constant__ float dev_volPlaneMax[DIM]; // bounding box planes maximum
__constant__ int dev_lastDetElement; 	 // number of last detector element

// __device__ variables are accessable from any thread and block as long as the programm is running

// error checking for GPU operations
#define gpuErrChk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// this kernels (__device__) can only be called from the device
__device__ inline float* max(float* vec, int length)
{
	auto max = vec;
	#pragma unroll
	for(auto i = 1; i < length; ++i)
	{
		if(*max < *(vec+i))
		{
			max = vec + i;
		}
	}
	return max;
}
__device__ inline float* min(float* vec, int length)
{
	auto min = vec;
#pragma unroll
	for (auto i = 1; i < length; ++i)
	{
		if (*min > *(vec + i))
		{
			min = vec + i;
		}
	}
	return min;
}

// this kernels (__global__) can be called from host and device
// initialize the rays between source and detector elements
__global__ void jacobsenRays(
	float* pts,
	float* projection
)
{
	// idenfifier of a single ray == threads identifier
	// (block index within grid) * (dimensions of block) + thread index within block
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// variables for each ray
	float t_ray[3], iMin[3], iMax[3];

	// check if the ray is the last element that needs to be computed
	if(idx < dev_lastDetElement)
	{
		printf("INITRAYS\n");
		#pragma unroll
		for(auto i = 0; i < 3; ++i)
		{
			// compute ray from source to detector
			// get detector point from linear array, source is the same for all rays -> representation of a ray
			t_ray[i] = pts[i*dev_lastDetElement + idx] - dev_source[i];
			// compute the parametrization when the ray hits and leaves the volume
			if (t_ray[i] != 0)
			{
				t_ray[i] += RAY_TOLERANCE;
				iMin[i] = (dev_volPlaneMin[i] - dev_source[i]) / t_ray[i];
				iMax[i] = (dev_volPlaneMax[i] - dev_source[i]) / t_ray[i];
			}
			else
			{
				t_ray[i] += RAY_TOLERANCE;
				iMin[i] = 0;
				iMax[i] = 1;
			}
		}
		printf("dev_source: %.2f,%.2f,%.2f\n", dev_source[0], dev_source[1], dev_source[2]);
		printf("pts: %.2f,%.2f,%.2f\n", pts[idx], pts[1*dev_lastDetElement+idx], pts[2*dev_lastDetElement+idx]);
		printf("t_ray: %.2f,%.2f,%.2f\n", t_ray[0], t_ray[1], t_ray[2]);
		//printf("iMin: %.2f,%.2f,%.2f\n", iMin[0], iMin[1], iMin[2]);
		//printf("iMax: %.2f,%.2f,%.2f\n", iMax[0], iMax[1], iMax[2]);

		// find the largest distance that can be run before hitting the volume
		float tmin[] = { fmin(iMin[0],iMax[0]),fmin(iMin[1],iMax[1]),fmin(iMin[2],iMax[2]),0.0f };
		//printf("tmin: %.2f,%.2f,%.2f,%.2f\n", tmin[0], tmin[1], tmin[2], tmin[3]);
		auto ti_ptr = max(tmin, 4);
		printf("ti: %.2f\n", *ti_ptr);
		// compute the smallest distance the ray have to run before leaving the volume
		float tmax[] = { fmax(iMin[0],iMax[0]),fmax(iMin[1],iMax[1]),fmax(iMin[2],iMax[2]),1.0f };
		//printf("tmax: %.2f,%.2f,%.2f,%.2f\n", tmax[0], tmax[1], tmax[2], tmax[3]);
		auto tf_ptr = min(tmax, 4);
		printf("tf: %.2f\n", *tf_ptr);

		float direction[3];
		#pragma unroll
		// save the variables for incremental kernel
		for (auto i = 0; i < 3; ++i)
			direction[i] = (pts[i] < dev_source[i])*(-2.0f) + 1.0f;
		printf("direction: %.2f, %.2f, %.2f\n", direction[0], direction[1], direction[2]);

		// compute the minimum and maximum intersection plane for each dimension (ti,tf)
		int p_min[3], p_max[3];
		float alpha[3];
		#pragma unroll
		for (auto i = 0; i < 3; ++i)
		{
			if (direction[i] > 0)
			{
				// if ray enters from origin site
				if (*ti_ptr == tmin[i])
					p_min[i] = 1;
				else
					p_min[i] = ceil(dev_source[i] + *ti_ptr *  t_ray[i] - dev_volumeOrigin[i]);
				if (*tf_ptr == tmax[i])
					p_max[i] = dev_dimension[i] - 1;
				else
					p_max[i] = floor(dev_source[i] + *tf_ptr *  t_ray[i] - dev_volumeOrigin[i]);

				alpha[i] = (dev_volumeOrigin[i] + p_min[i] - dev_source[i]) / t_ray[i];
			}
			else
			{
				// if ray enters from back site of volume
				if (*ti_ptr == tmin[i])
					p_max[i] = dev_dimension[i] - 2;
				else
					p_max[i] = floor(dev_source[i] + *ti_ptr *  t_ray[i] - dev_volumeOrigin[i]);
				if (*tf_ptr == tmax[i])
					p_min[i] = 0;
				else
					p_min[i] = ceil(dev_source[i] + *tf_ptr *  t_ray[i] - dev_volumeOrigin[i]);

				alpha[i] = (dev_volumeOrigin[i] + p_max[i] - dev_source[i]) / t_ray[i];
			}
		}
		//printf("p_min: %i, %i, %i\n", p_min[0], p_min[1], p_min[2]);
		//printf("p_max: %i, %i, %i\n", p_max[0], p_max[1], p_max[2]);
		
		// number of crossed planes regardless of dimension
		auto n_planes = p_max[0] - p_min[0] + 1 + p_max[1] - p_min[1] + 1 + p_max[2] - p_min[2] + 1;
		//printf("n_planes: %i\n", n_planes);
		printf("=============================================\n");

		// first intersections with x,y,z plane
		int index[3];
		auto alpha_min = min(alpha, 3);
		#pragma unroll
		for(auto i = 0; i < 3; ++i)
		{
			index[i] = floor(dev_source[i] + ((*alpha_min + *ti_ptr) * 0.5f) * t_ray[i] - dev_volumeOrigin[i]);
		}
		//printf("index: %i, %i, %i\n", index[0], index[1], index[2]);
				
		// incrementation of ray
		// update step
		float alpha_update[3] = { 1/fabs(t_ray[0]), 1/fabs(t_ray[1]), 1/fabs(t_ray[2]) };
		auto alpha_current = *ti_ptr;
		auto raysum = .0f;
		float voxlen, mu;

		printf("a_xyz: %.2f, %.2f, %.2f\n", alpha[0], alpha[1], alpha[2]);
		printf("a_cur: %.2f\n", alpha_current);
		do
		{
			// get voxel value and the intersection length for updating raysum
			mu = tex3D(dev_tex, index[0], index[1], index[2]);
			// printf("mu: %.2f\n", mu);

			if( alpha[0] < alpha[1] && alpha[0] < alpha[2])
			{
				voxlen = alpha[0] - alpha_current;
				index[0] += direction[0];
				alpha_current = alpha[0];
				alpha[0] += alpha_update[0];
				
			}
			else if (alpha[1] < alpha[2])
			{
				voxlen = alpha[1] - alpha_current;
				index[1] += direction[1];
				alpha_current = alpha[1];
				alpha[1] += alpha_update[1];
			}
			else
			{
				voxlen = alpha[2] - alpha_current;
				index[2] += direction[2];
				alpha_current = alpha[2];
				alpha[2] += alpha_update[2];
			}

			// accumulate absorption
			raysum += voxlen*mu;
			
			printf("voxlen: %.2f\n", voxlen);
			printf("alpha_current: %.2f\n", alpha_current);
			printf("index: %i, %i, %i\n", index[0], index[1], index[2]);
			
		} while (alpha_current < *tf_ptr);

		// normalize to length of ray
		printf("ray_length: %.2f\n", sqrt(pow(t_ray[0], 2.f) + pow(t_ray[1], 2.f) + pow(t_ray[2], 2.f)));
		projection[idx] = raysum * sqrt(pow(t_ray[0], 2.f) + pow(t_ray[1], 2.f) + pow(t_ray[2], 2.f));
		printf("projection: %.2f\n", projection[idx]);

		// synchronise the threads -> maybe not necessary
		__syncthreads();
	}
} // end jacobsen kernel


CudaTest::CudaTest() {

	// get number of GPU devices
	auto numberOfGPUDevices = 0;
	gpuErrChk(cudaGetDeviceCount(&numberOfGPUDevices));
	if(numberOfGPUDevices > 0 )
	{
		showDevices(numberOfGPUDevices);
		dev_ID = 0;
		dev_State = AVAILABLE;
	}
	else
	{
		printf("No CUDA device available.");
		dev_State = NOTAVAILABLE;
	}
	
	// initialize the array with null pointer to avoid exception while freeing
	dev_Array = nullptr;

	// initialize the variables shared between ost and device
	for(auto i = 0; i < DIM; ++i)
	{
		host_dimension[i] = 0;
	}

}

CudaTest::~CudaTest()
{
	// unbind the texture before freeing the memory to avoid undefined behavior
	gpuErrChk(cudaUnbindTexture(dev_tex));
	// free device array
	gpuErrChk(cudaFreeArray(dev_Array));
}

// print device properties for all devices
void CudaTest::showDevices(int numberOfGPUDevices)
{
	for (auto i = 0; i < numberOfGPUDevices; i++)
	{
		cudaDeviceProp prop;
		gpuErrChk(cudaGetDeviceProperties(&prop, i));
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}

// copy voxel to the internal memory on the device
void CudaTest::copyVolumeToGPU(const int* dimension, const float* origin, const VoxelType* data)
{
	// start time measurement
	auto start = hr_clock::now();

	// cudaExtent contains the dimensions of the volume that is put on the GPU memory
	auto volumeSize = make_cudaExtent(dimension[0], dimension[1], dimension[2]);
	// describes the texel format (in this case it's float)
	auto channelDes = cudaCreateChannelDesc<VoxelType>();
	// allocate memory on the GPU for the volume data
	gpuErrChk(cudaMalloc3DArray(&dev_Array, &channelDes, volumeSize));
	// copy parameters between CPU and GPU memory
	cudaMemcpy3DParms copyVolumeParams = { 0 };         // the initialization with 0 is recommended
	copyVolumeParams.kind     = cudaMemcpyHostToDevice; // copying direction (host->device)
	copyVolumeParams.srcPtr   = make_cudaPitchedPtr((void*)data, volumeSize.width * sizeof(VoxelType), volumeSize.width, volumeSize.height); // pointer with row,slice pitch
	copyVolumeParams.dstArray = dev_Array;              // destination array for copy process
	copyVolumeParams.extent = volumeSize;				// size of the copied memory
	
	// start the copy process
	gpuErrChk(cudaMemcpy3D(&copyVolumeParams));

	// set nearest neighbor interpolation
	dev_tex.filterMode = cudaFilterModePoint;
	// set border priorities (clamp = extent the last voxel value around the border)
	dev_tex.addressMode[0] = cudaAddressModeClamp;
	dev_tex.addressMode[1] = cudaAddressModeClamp;
	dev_tex.addressMode[2] = cudaAddressModeClamp;
	//set normalization of textures (normalized from [0,1])
	dev_tex.normalized = false;
	// bind copied data to texture memory (the error here is the only way)
	gpuErrChk(cudaBindTextureToArray(dev_tex, dev_Array, channelDes));


	// connect the host and device variables and copy the values to device
	float host_volPlaneMin[DIM];
	float host_volPlaneMax[DIM];
	for (auto i = 0; i < DIM; i++) {
		host_dimension[i] = dimension[i];
		host_volPlaneMin[i] = origin[i];
		host_volPlaneMax[i] = origin[i] + host_dimension[i];
	}

	gpuErrChk(cudaMemcpyToSymbol(dev_dimension, host_dimension, DIM*sizeof(int)));
	// copy the volume position to constant device memory
	gpuErrChk(cudaMemcpyToSymbol(dev_volPlaneMin, host_volPlaneMin, DIM * sizeof(float)));
	gpuErrChk(cudaMemcpyToSymbol(dev_volPlaneMax, host_volPlaneMax, DIM * sizeof(float)));

	dev_State = DATALOADED;
	// stop timing until after execution and output the duration in microseconds
	cudaDeviceSynchronize();
	auto stop = hr_clock::now();
	timing(stop-start, "Copy volume to device");
}

// copy projector to device memory
void CudaTest::copyProjectorToGPU(
	const float* srcPosition,
	const int	 detectorElements,
	const float* detElX,
	const float* detElY,
	const float* detElZ,
	float* detArray
){
	printf("pts_test: %.2f,%.2f,%.2f\n", *detElX, *detElY, *detElZ);
	// start timing
	auto start = hr_clock::now();

	// copy projection variables to device
	gpuErrChk(cudaMemcpyToSymbol(dev_source, srcPosition, DIM*sizeof(float)));
	gpuErrChk(cudaMemcpyToSymbol(dev_lastDetElement, &detectorElements, sizeof(int)));

	// copy detector element positions to device
	float* dev_projection; // detector value array
	float* dev_pts; // array for x,y,z components of detector elements
	gpuErrChk(cudaMalloc((void**)&dev_projection, detectorElements * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&dev_pts,        DIM * detectorElements * sizeof(float)));

	// copy detector element positions to device
	gpuErrChk(cudaMemcpy(dev_pts,                        detElX, detectorElements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(dev_pts + detectorElements,     detElY, detectorElements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(dev_pts + 2 * detectorElements, detElZ, detectorElements * sizeof(float), cudaMemcpyHostToDevice));

	// stop timing
	cudaDeviceSynchronize();
	auto stop = hr_clock::now();
	timing(stop - start, "Loading geometry to device");

	dev_State = DATAANDGEOMETRYLOADED;

	// setup the computing threads and blocks -> TODO calculate the optimum
	auto threadsPerBlock = 256;
	auto numberBlocks = detectorElements / threadsPerBlock + 1;

	// start kernel functions for computing projections
	start = hr_clock::now();
	jacobsenRays <<< numberBlocks, threadsPerBlock >>>(dev_pts, dev_projection);
	cudaDeviceSynchronize();
	stop = hr_clock::now();
	timing(stop - start, "Computing the projections");


	// copy projection back to host
	start = hr_clock::now();
	gpuErrChk(cudaMemcpy(detArray, dev_projection, detectorElements * sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	stop = hr_clock::now();
	timing(stop - start, "Retrieving detector values");

	// check for errors
	gpuErrChk(cudaDeviceSynchronize());
	gpuErrChk(cudaGetLastError());

	// cleanup
	gpuErrChk(cudaFree(dev_projection));
	gpuErrChk(cudaFree(dev_pts));
}

// timing function and output
void CudaTest::timing(std::chrono::duration<long long,std::nano> dur, std::string txt) const
{
	auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	printf("%s: %lld mcrs.\n",txt.c_str(), microseconds);
}
