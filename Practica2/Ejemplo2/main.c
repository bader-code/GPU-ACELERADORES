#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "my_ocl.h"

#include "md5.h"

#define MAX_DIGIT_LEN 10


double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}

int hex2int(char ch){
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    return -1;
}

// Function generate msg, from a value it=(0, basis[nchars])
void gen_msg(char msg[], int nchars, char cset[], int basis[], int it)
{

	msg[nchars+1]='\0';
	for(int j=0; j<=nchars; j++)
	{
		if (j>0)
			msg[nchars-j] = cset[(it%basis[j])/basis[j-1]];
		else
			msg[nchars-j] = cset[it%basis[j]];

	}
}

int force_brute(char msg_cracked[], int nchars, char cset[], int h0_md5, int h1_md5, int h2_md5, int h3_md5, int basis[]){
	char msg[MAX_DIGIT_LEN];

	int nostop = 1;
	int h0, h1, h2, h3;

	for (int nchar=0; nchar<nchars; nchar++)
	{
		printf("Trying m5d with %d characters!!\n", nchar+1);

		for(int ichar=0; ichar<basis[nchar]; ichar++)
		{
			gen_msg(msg, nchar, cset, basis, ichar);

			md5(&h0, &h1, &h2, &h3, msg, strlen(msg));
//			printf("value=%s(%d) %x %x %x %x\n", msg, strlen(msg), h0, h1, h2, h3);

            nostop=(h0==h0_md5)&(h1==h1_md5)&(h2==h2_md5)&(h3==h3_md5); //h0, h1, h2, h3 comparison
			if(nostop){
				strcpy(msg_cracked, msg);
				return(1);
			}

		}
	}

	return(0);
} 




int main(int argc, char **argv) {

	char cset[] = "abcdefghijklmnopqrstuvwxyz 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\0";
	char msg_cracked[MAX_DIGIT_LEN];
	char tmp[8];
	char *md5_hash = argv[1];


	size_t len = strlen(md5_hash);
	int found;

	//Variables para calcular el tiempo
	double t0, t1;
	double cpu_time_used = 0.0;

	if (argc != 3){ printf("./exec md5_hash[32chars] [c,g]\n"); exit(-1); }

	if (len != 32){ printf("./exec md5_hash[32chars]\n"); exit(-1); }

	int h, h0_md5, h1_md5, h2_md5, h3_md5;
	for(int i=0; i<32; i+=8){
		h=0;
		for (int j=0; j<8; j++){ 
			char byte = md5_hash[i+j];
			char hex = hex2int(byte);
			h+=hex<<((7-j)*4);
		}
		if (i==0) h0_md5=h;
		if (i==8) h1_md5=h;
		if (i==16) h2_md5=h;
		if (i==24) h3_md5=h;
	}

	int basis[MAX_DIGIT_LEN];

	int acc=1;
	for(int i=0; i<MAX_DIGIT_LEN; i++){
		acc *= strlen(cset);
		basis[i]=acc;
	}


	switch (argv[2][0]) {
		case 'c':
			t0 = get_time();
			found = force_brute(msg_cracked, 5, cset, h0_md5, h1_md5, h2_md5, h3_md5, basis);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
			t0 = get_time();
			remove_noiseOCL(msg_cracked, 5, cset, h0_md5, h1_md5, h2_md5, h3_md5, basis);
			t1 = get_time();
			printf("OCL Exection time %f ms.\n", t1-t0);
			break;
		default:
			printf("Not Implemented yet!!\n");
	}

	if (found)
		printf("md5=%s corresponds to msg=%s\n", md5_hash, msg_cracked);
	else
		printf("Not found!!!!\n");

}

