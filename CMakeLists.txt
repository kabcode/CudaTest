cmake_minimum_required(VERSION 3.0)
project(CudaTest)

# provide CMAKE variables
message( STATUS "CMAKE_SOURCE_DIR:         " ${CMAKE_SOURCE_DIR} )
message( STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_BINARY_DIR} )
message( STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH} )
message( STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX} )

######################
### FIND LIBRARIES ###
######################
# find the CUDA package
find_package(CUDA REQUIRED)

#####################
### ADD OWN FILES ###
#####################
# collect CUDA files
set(CUDA_SRCS
	cudaTest.cu
)
set(CUDA_HDRS
	cudaTest.h
) 

# collect CPP files (source and header)
set(CPP_SRCS
)
	
set(CPP_HDRS
	CImg.h
)

# create the executable
cuda_add_executable(main main.cpp
	${CUDA_SRCS}
	${CUDA_HDRS}
	${CPP_SRCS}
	${CPP_HDRS}
)