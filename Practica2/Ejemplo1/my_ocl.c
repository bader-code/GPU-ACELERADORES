#include <stdio.h>
#include <stdlib.h>
#include "my_ocl.h"
#include "CL/cl.h"

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define BLOCKSZ 8


/* From common.c */
extern double getMicroSeconds();
extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);
extern float *getmemory1D( int nx );
extern int check(float *GPU, float *CPU, int n);


void remove_noiseOCL(float *im, float *image_out,  float thredshold, int window_size, int height, int width)
{
	/** ********************* **/
	/** OpenCL host variables **/
	/** ********************* **/
	cl_uint numPlatforms;					 
	cl_int err;								 
	cl_platform_id *Platform;				
	cl_device_id deviceId;     		         
	cl_context context;       				
	cl_command_queue commands;     		    
	cl_program program;       				 
	cl_kernel kernel;       				 
	
	/** ***************************************** **/
	/** variables used to read kernel source file **/
	/** ***************************************** **/
	FILE *fp;		   
	long filelen;	   
	long readlen;	   
	char *kernel_src;  

	/** *************** **/
	/** Other Variables **/
	/** *************** **/
	int i;
	cl_float *img;		
    cl_float *imgRN;   

	const size_t global[2] = {width, height};   
	const size_t local[2] = {BLOCKSZ,BLOCKSZ}; 
	
        
    

	/** *************** **/
	/** read the kernel **/
	/** *************** **/
	fp = fopen("kernelRemoveNoise.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen){
		printf("error reading file\n");
		exit(1);
	}
	
	kernel_src[filelen]='\0';

	/** Find number of platforms **/
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0){
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	/** Get all platforms **/
	Platform = (cl_platform_id *) malloc(sizeof(numPlatforms));
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0){
        printf("Error: Failed to get the platform!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }

	/** Secure a GPU **/
    for (i = 0; i < numPlatforms; i++){
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &deviceId, NULL);
        if (err == CL_SUCCESS){
            break;
        }
    }
	if (deviceId == NULL){
        printf("Error: Failed to create a device group!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }
    err = output_device_info(deviceId);

	/** Create a compute context **/
    context = clCreateContext(0, 1, &deviceId, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

	/** Create a command queue **/
    commands = clCreateCommandQueue(context, deviceId, 0, &err);
    if (!commands){
        printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

	/** Create the compute program from the source buffer **/
    program = clCreateProgramWithSource(context, 1, (const char **) & kernel_src, NULL, &err);
    if (!program){
        printf("Error: Failed to create compute program!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

	/** Build the program **/
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

	/** Create the compute kernel from the program **/
    kernel = clCreateKernel(program, "kernelRemoveNoise", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

	/** create buffer objects to input and output args of kernel function **/
	img   = clCreateBuffer(context, CL_MEM_READ_ONLY, (sizeof(float)*height * width), NULL, NULL);
	imgRN = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (sizeof(float)*height * width), NULL, NULL);

	/** Write a and b vectors into compute device memory **/
    err = clEnqueueWriteBuffer(commands, img, CL_TRUE, 0, (sizeof(float)*height * width), im, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to write h_a to source array!\n%s\n", err_code(err));
        exit(1);
    }

	/** set the kernel arguments **/
	if ( clSetKernelArg(kernel, 0, sizeof(cl_float *), &img)   ||
         clSetKernelArg(kernel, 1, sizeof(cl_float *), &imgRN) ||
         clSetKernelArg(kernel, 2, (((1 + BLOCKSZ + 1)*(1 + BLOCKSZ + 1)) * sizeof(cl_float *)), NULL) ||
		 clSetKernelArg(kernel, 3, sizeof(cl_float), &thredshold) ||
         clSetKernelArg(kernel, 4, sizeof(cl_int), &window_size)  ||
		 clSetKernelArg(kernel, 5, sizeof(cl_int), &height)  	  ||
		 clSetKernelArg(kernel, 6, sizeof(cl_int), &width) != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	double t0 = getMicroSeconds();
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
	double t1 = getMicroSeconds();
	printf("\nThe kernel ran in %lf seconds\n",(t1-t0)/1000000);
	if (err != CL_SUCCESS){	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	/** wait for the command to finish **/
	clFinish(commands);

	err = clEnqueueReadBuffer(commands, imgRN, CL_TRUE, 0, (sizeof(float)* height * width), image_out, 0, NULL, NULL);
	if (err != CL_SUCCESS){	
		printf("Error enqueuing read buffer command. Error Code=%d\n",err);
		exit(1);
	}
	
	/** clean up **/
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	free(kernel_src);
	free(Platform);
	clReleaseMemObject(img);
	clReleaseMemObject(imgRN);

}
