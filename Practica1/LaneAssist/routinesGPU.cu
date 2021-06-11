#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"
#define DEG2RAD 0.017453f
#define BLOCKSZ 16
#define PADDING 1


__global__ void NRcanny(uint8_t *im, float *NR, int height, int width){

	unsigned int i = (blockIdx.y * blockDim.y) + threadIdx.y, j = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int ii = (threadIdx.y) + 2, jj = (threadIdx.x) + 2;

	__shared__ uint8_t sh_im[2 + BLOCKSZ + 2 + PADDING][2 + BLOCKSZ + 2];

	if(i < height && j < width){
		sh_im[jj][ii] = im[(i * width) + j];
	}


	if((ii == 2 || ii == 3) && i >= 2){ 
		sh_im[jj][ii - 2] = im[((i - 2) * width) + j];
		if((jj == 2 || jj == 3) && j >= 2){
			sh_im[jj - 2][ii - 2] = im[((i - 2) * width) + (j - 2)]; 
		}
	}
	if((jj == 2 || jj == 3) && j >= 2){
		sh_im[jj - 2][ii] = im[(i * width) + (j - 2)];
		if((ii == BLOCKSZ + 1 || ii == BLOCKSZ) && i < (height - 2)){
			sh_im[jj - 2][ii + 2] = im[((i + 2) * width) + (j - 2)];
		}
	}
	if((ii == BLOCKSZ + 1 || ii == BLOCKSZ) && i < (height - 2)){ 
		sh_im[jj][ii + 2] = im[((i + 2) * width) + j];
		if((jj == BLOCKSZ + 1 || jj == BLOCKSZ) && j < (width - 2)){
			sh_im[jj + 2][ii + 2] = im[((i + 2) * width) + (j + 2)];
		}
	}
	if((jj == BLOCKSZ + 1 || jj == BLOCKSZ) && j < (width - 2)){
		sh_im[jj + 2][ii] = im[(i * width) + (j + 2)];
		if((ii == 2 ||ii == 3) && i >= 2){
			sh_im[jj + 2][ii - 2] = im[((i - 2) * width) + (j + 2)]; 
		}
	}

	__syncthreads();

	if(i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)){
		// Noise reduction
		NR[(i * width) + j] = 
			 (2.0*sh_im[jj - 2][ii - 2] +  4.0*sh_im[jj - 1][ii - 2] +  5.0*sh_im[jj    ][ii - 2] +  4.0*sh_im[jj + 1][ii - 2] + 2.0*sh_im[jj + 2][ii - 2]
			+ 4.0*sh_im[jj - 2][ii - 1] +  9.0*sh_im[jj - 1][ii - 1] + 12.0*sh_im[jj    ][ii - 1] +  9.0*sh_im[jj + 1][ii - 1] + 4.0*sh_im[jj + 2][ii - 1]
			+ 5.0*sh_im[jj - 2][ii    ] + 12.0*sh_im[jj - 1][ii    ] + 15.0*sh_im[jj    ][ii    ] + 12.0*sh_im[jj + 1][ii    ] + 5.0*sh_im[jj + 2][ii    ]
			+ 4.0*sh_im[jj - 2][ii + 1] +  9.0*sh_im[jj - 1][ii + 1] + 12.0*sh_im[jj    ][ii + 1] +  9.0*sh_im[jj + 1][ii + 1] + 4.0*sh_im[jj + 2][ii + 1]
			+ 2.0*sh_im[jj - 2][ii + 2] +  4.0*sh_im[jj - 1][ii + 2] +  5.0*sh_im[jj    ][ii + 2] +  4.0*sh_im[jj + 1][ii + 2] + 2.0*sh_im[jj + 2][ii + 2])
			/159.0;
			
	}
}
__global__ void Gcanny(float *G, float *NR, float *phi, int height, int width){
	
	unsigned int i = (blockIdx.y * blockDim.y) + threadIdx.y, j = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int ii = (threadIdx.y) + 2, jj = (threadIdx.x) + 2;

	__shared__ float sh_NR[2 + BLOCKSZ + 2 + PADDING][2 + BLOCKSZ + 2];

	float Gy, Gx, phi_thread;
	float PI = 3.141593;

	if(i < height && j < width){
		sh_NR[jj][ii] = NR[(i * width) + j];
	}


	if((ii == 2 || ii == 3) && i >= 2){ 
		sh_NR[jj][ii - 2] = NR[((i - 2) * width) + j];
		if((jj == 2 || jj == 3) && j >= 2){
			sh_NR[jj - 2][ii - 2] = NR[((i - 2) * width) + (j - 2)]; 
		}
	}
	if((jj == 2 || jj == 3) && j >= 2){
		sh_NR[jj - 2][ii] = NR[(i * width) + (j - 2)];
		if((ii == BLOCKSZ + 1 || ii == BLOCKSZ) && i < (height - 2)){
			sh_NR[jj - 2][ii + 2] = NR[((i + 2) * width) + (j - 2)];
		}
	}
	if((ii == BLOCKSZ + 1 || ii == BLOCKSZ) && i < (height - 2)){ 
		sh_NR[jj][ii + 2] = NR[((i + 2) * width) + j];
		if((jj == BLOCKSZ + 1 || jj == BLOCKSZ) && j < (width - 2)){
			sh_NR[jj + 2][ii + 2] = NR[((i + 2) * width) + (j + 2)];
		}
	}
	if((jj == BLOCKSZ + 1 || jj == BLOCKSZ) && j < (width - 2)){
		sh_NR[jj + 2][ii] = NR[(i * width) + (j + 2)];
		if((ii == 2 ||ii == 3) && i >= 2){
			sh_NR[jj + 2][ii - 2] = NR[((i - 2) * width) + (j + 2)]; 
		}
	}

	__syncthreads();

	if(i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)){
		// Intensity gradient of the image
		Gx = 
			 (1.0*sh_NR[jj - 2][ii - 2] +  2.0*sh_NR[jj - 1][ii - 2] +  (-2.0)*sh_NR[jj + 1][ii - 2] + (-1.0)*sh_NR[jj + 2][ii - 2]
	   		+ 4.0*sh_NR[jj - 2][ii - 1] +  8.0*sh_NR[jj - 1][ii - 1] +  (-8.0)*sh_NR[jj + 1][ii - 1] + (-4.0)*sh_NR[jj + 2][ii - 1]
	   		+ 6.0*sh_NR[jj - 2][ii    ] + 12.0*sh_NR[jj - 1][ii    ] + (-12.0)*sh_NR[jj + 1][ii    ] + (-6.0)*sh_NR[jj + 2][ii    ]
	   		+ 4.0*sh_NR[jj - 2][ii + 1] +  8.0*sh_NR[jj - 1][ii + 1] +  (-8.0)*sh_NR[jj + 1][ii + 1] + (-4.0)*sh_NR[jj + 2][ii + 1]
	  		+ 1.0*sh_NR[jj - 2][ii + 2] +  2.0*sh_NR[jj - 1][ii + 2] +  (-2.0)*sh_NR[jj + 1][ii + 2] + (-1.0)*sh_NR[jj + 2][ii + 2]);


   		Gy = 
			((-1.0)*sh_NR[jj - 2][ii - 2] + (-4.0)*sh_NR[jj - 1][ii - 2] +  (-6.0)*sh_NR[jj    ][ii - 2] + (-4.0)*sh_NR[jj + 1][ii - 2] + (-1.0)*sh_NR[jj + 2][ii - 2]
	   		+(-2.0)*sh_NR[jj - 2][ii - 1] + (-8.0)*sh_NR[jj - 1][ii - 1] + (-12.0)*sh_NR[jj    ][ii - 1] + (-8.0)*sh_NR[jj + 1][ii - 1] + (-2.0)*sh_NR[jj + 2][ii - 1]
	   		+   2.0*sh_NR[jj - 2][ii + 1] +    8.0*sh_NR[jj - 1][ii + 1] +    12.0*sh_NR[jj    ][ii + 1] +    8.0*sh_NR[jj + 1][ii + 1] +    2.0*sh_NR[jj + 2][ii + 1]
	   		+   1.0*sh_NR[jj - 2][ii + 2] +    4.0*sh_NR[jj - 1][ii + 2] +     6.0*sh_NR[jj    ][ii + 2] +    4.0*sh_NR[jj + 1][ii + 2] +    1.0*sh_NR[jj + 2][ii + 2]);
		
		G[i*width+j] = sqrtf((Gx*Gx)+(Gy*Gy));	//G = √Gx²+Gy²
		phi_thread = atan2f(fabs(Gy),fabs(Gx));

		if(fabs(phi_thread)<=PI/8 ) phi[i*width+j] = 0;
		else if (fabs(phi_thread)<= 3*(PI/8)) phi[i*width+j] = 45;
		else if (fabs(phi_thread) <= 5*(PI/8)) phi[i*width+j] = 90;
		else if (fabs(phi_thread) <= 7*(PI/8)) phi[i*width+j] = 135;
		else phi[i*width+j] = 0;
	}

	
}
__global__ void PEDGEcanny(float *G, uint8_t *imEdge, float *phi, int height, int width, float level){

	unsigned int i = (blockIdx.y * blockDim.y) + threadIdx.y, j = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int ii = (threadIdx.y) + 3, jj = (threadIdx.x) + 3;
	int iii = 0, jjj = 0;

	__shared__ float sh_G[3 + BLOCKSZ + 3 + PADDING][3 + BLOCKSZ + 3];

	float phi_thread, G_thread, lowthres = (level/2), hithres = 2*(level);
	uint8_t pedge = 0;

	if(i < height && j < width){
		sh_G[jj][ii] = G[(i * width) + j];
	}

	if(ii == 3 && i >= 3){
		sh_G[jj][ii - 1] = G[((i - 1) * width) + j]; 
		if(jj == 3 && j >= 3){
			sh_G[jj - 1][ii - 1] = G[((i - 1) * width) + (j - 1)];
		}
	}
	if(jj == 3 && j >= 3){
		sh_G[jj - 1][ii] = G[(i * width) + (j - 1)]; 
		if(ii == BLOCKSZ + 2 && i < (height - 3)){
			sh_G[jj - 1][ii + 1] = G[((i + 1) * width) + (j - 1)]; 

		}
	}
	if(ii == BLOCKSZ + 2 && i < (height - 3)){
		sh_G[jj][ii + 1] = G[((i + 1) * width) + j]; 
		if(jj == BLOCKSZ + 2 && j < (width - 3)){
			sh_G[jj + 1][ii + 1] = G[((i + 1) * width) + (j + 1)]; 
		}
	}
	if(jj == BLOCKSZ + 2 && j < (width - 3)){
		sh_G[jj + 1][ii] = G[(i * width) + (j + 1)]; 
		if(ii == 3 && i >= 3){
			sh_G[jj + 1][ii - 1] = G[((i - 1) * width) + (j + 1)];
		}
	}

	__syncthreads();

	if(i >= 3 && j >= 3 && i < (height - 3) && j < (width - 3)){
		phi_thread = phi[i*width+j];
		G_thread = sh_G[jj][ii];
		if(phi_thread == 0){
			if(G_thread > sh_G[jj + 1][ii] && G_thread > sh_G[jj - 1][ii]) //edge is in N-S
			pedge = 1;

		} else if(phi_thread == 45) {
			if(G_thread > sh_G[jj + 1][ii + 1] && G_thread > sh_G[jj - 1][ii - 1]) // edge is in NW-SE
			pedge= 1;

		} else if(phi_thread == 90) {
			if(G_thread > sh_G[jj][ii + 1] && G_thread > sh_G[jj][ii - 1]) //edge is in E-W
			pedge = 1;

		} else if(phi_thread == 135) {
			if(G_thread > sh_G[jj - 1][ii + 1] && G_thread > sh_G[jj + 1][ii - 1]) // edge is in NE-SW
			pedge = 1;
		}

		if(G_thread > hithres && pedge ){ imEdge[i*width+j] = 255; }
		else if(pedge && G_thread >= lowthres && G_thread < hithres){
			// check neighbours 3x3
			for (iii = -1; iii <= 1; iii++){
				for (jjj = -1; jjj <= 1; jjj++){
					if (sh_G[jj + iii][ii+jjj] > hithres) {imEdge[i*width+j] = 255; iii = 2; jjj = 2;}
				}
			}
		}
	}

}


void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  
				//local maxima
				if(max == accumulators[(rho*accu_width) + theta]){
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

__global__ void accumulatorshoughtransform(uint8_t *im, uint32_t *accumulators, float *sin_table, float *cos_table, int width, int height,
	float hough_h, float center_x, float center_y){
		
		int j = (blockIdx.x * blockDim.x) + threadIdx.x, i = (blockIdx.y * blockDim.y) + threadIdx.y;
		int ii = threadIdx.x, jj = threadIdx.y;

		__shared__ float sh_sin[180 + PADDING];
		__shared__ float sh_cos[180 + PADDING];

		if((ii * BLOCKSZ) + jj < 180){
			sh_sin[(ii * BLOCKSZ) + jj] = sin_table[(ii * BLOCKSZ) + jj];
			sh_cos[(ii * BLOCKSZ) + jj] = cos_table[(ii * BLOCKSZ) + jj];
		}
		__syncthreads();

		int theta = 0;
		float rho = 0;

		if(i < height && j < width){
			if(im[ (i*width) + j] > 250){
				for(theta = 0; theta < 180; theta++) {  
					rho = ( ((float)j - center_x) * sh_cos[theta]) + (((float)i - center_y) * sh_sin[theta]);
					atomicAdd(&accumulators[(int)((round(rho + hough_h) * 180.0)) + theta], 1);
				} 
			}
		}
}


void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	int threshold;
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	float center_x = width/2.0; 
	float center_y = height/2.0;

	/**			canny		**/
	uint8_t *im_GPU, *imEdge_GPU;
	float   *NR_GPU, *G_GPU, *phi_GPU, *sin_table_GPU, *cos_table_GPU;
	uint32_t *accum_GPU;

	//Reserva de memoria
	cudaMalloc((void**)&im_GPU,  sizeof(uint8_t) * width * height);
	cudaMalloc((void**)&NR_GPU,  sizeof(float)   * width * height);
	cudaMalloc((void**)&G_GPU,   sizeof(float)   * width * height);
	cudaMalloc((void**)&phi_GPU, sizeof(float)   * width * height);
	cudaMalloc((void**)&imEdge_GPU, sizeof(uint8_t) * width * height);

	dim3 dimBlock(BLOCKSZ,BLOCKSZ);
	dim3 dimGrid((width / dimBlock.x) + 1, (height / dimBlock.y) + 1);

	cudaMemcpy(im_GPU,im, (sizeof(uint8_t) * width * height), cudaMemcpyHostToDevice);
	NRcanny<<<dimGrid,dimBlock>>>(im_GPU, NR_GPU, height, width);
	cudaThreadSynchronize();

	Gcanny<<<dimGrid,dimBlock>>>(G_GPU, NR_GPU, phi_GPU, height, width);
	cudaThreadSynchronize();

	PEDGEcanny<<<dimGrid,dimBlock>>>(G_GPU, imEdge_GPU, phi_GPU, height, width, 1000.0f);
	cudaThreadSynchronize();

	/** 	hough transform 	**/
	//Reserva de memoria
	cudaMalloc((void**)&accum_GPU,  sizeof(uint32_t) * accu_width * accu_height);
	cudaMalloc((void**)&sin_table_GPU,  sizeof(float) * 180);
	cudaMalloc((void**)&cos_table_GPU,  sizeof(float) * 180);

	cudaMemset(accum_GPU, 0, (sizeof(uint32_t) * accu_width * accu_height));

	cudaMemcpy(cos_table_GPU,cos_table, (sizeof(float) * 180), cudaMemcpyHostToDevice);
	cudaMemcpy(sin_table_GPU,sin_table, (sizeof(float) * 180), cudaMemcpyHostToDevice);

	accumulatorshoughtransform<<<dimGrid,dimBlock>>>(imEdge_GPU, accum_GPU, sin_table_GPU, cos_table_GPU, width, height, hough_h, center_x, center_y);
	cudaThreadSynchronize();

	cudaMemcpy(accum, accum_GPU, (sizeof(uint32_t) * accu_width * accu_height), cudaMemcpyDeviceToHost);

	if (width>height) threshold = width/6;
	else threshold = height/6;

	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}
