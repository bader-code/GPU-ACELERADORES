#define BLOCKSZ 8
#define MAX_WINDOW_SIZE 5*5


float fabsf(float value){
	return (value > 0) ? value : ( -1 * value);
}

void buble_sort(float array[], int size){
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}

__kernel void kernelRemoveNoise(__global float * img, __global float * imgRN, __local float * sh_img, float thredshold, int window_size, int height, int width){
	
	unsigned int i = get_global_id(1), j = get_global_id(0);
	int ii,jj;
	__private float window[MAX_WINDOW_SIZE];

	if(i >= 1 && j >= 1 && i < (height - 1) && j < (width - 1)){
		for (ii = -1; ii <= 1; ii++){
			for (jj = -1; jj <= 1; jj++){
				window[((ii + 1) * window_size) + (jj+ 1)] = img[(i + ii)*width + (j + jj)];
			}
		}
		buble_sort(window, (window_size * window_size));
		int median =  window[((window_size*window_size) - 1) >> 1];
		if (fabsf((median - img[(i * width) + j]) / median) <= thredshold){
			imgRN[i*width + j] = img[i*width+j];
		}
		else{
			imgRN[(i * width) + j] = median;
		}
	}
}

 


	
