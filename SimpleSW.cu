#include<cuda.h>
#include<nvvm.h>
#include<unistd.h>
#include<malloc.h>

#include<iostream>
#include<iomanip>
#include"SimpleSW.h"
#include"Data.h"

// Lenght of each data
__constant__ int gcT_size;
__constant__ int gcP_size;

// Threshold of the SW algorithm
__constant__ int gcThre;

// Data of the query
__constant__ char gcP_seq[1024];

// Cost and Gain
__constant__ int gcMatch;
__constant__ int gcMiss;
__constant__ int gcExtend;
__constant__ int gcBegin;

enum{
	Zero,
	Diagonal,
	Vertical,
	Horizon,
};

using namespace std;

SimpleSW::SimpleSW(const Data& txt, const Data& ptn, int threshold){
	cout << "At the begining of the SW algorithm (Simple)" << endl;
	// Sieze check
	// TODO Max size
	if(ptn.size() > 1024 || ptn.size() * txt.size() > 1024 * 1024 * 1024){
		std::cout << "Too large size" << std::endl;
		return;
	}
	// Set value in constant memory
	int tsize = txt.size();
	int psize = ptn.size();
	cudaMemcpyToSymbol(gcT_size, &tsize, sizeof(int));
	cudaMemcpyToSymbol(gcP_size, &psize, sizeof(int));
	cudaMemcpyToSymbol(gcThre, &threshold, sizeof(int));
	cudaMemcpyToSymbol(gcP_seq, ptn.data(), sizeof(char) * ptn.size());
	// TODO Cost and gain
	int gain = 1;
	cudaMemcpyToSymbol(gcMatch, &gain, sizeof(int));
	gain = -1;
	cudaMemcpyToSymbol(gcMiss, &gain, sizeof(int));
	gain = -1;
	cudaMemcpyToSymbol(gcExtend, &gain, sizeof(int));
	gain = -3;
	cudaMemcpyToSymbol(gcBegin, &gain, sizeof(int));
	// Dynamic Programing part by call_DP
	call_DP(txt, ptn);
	std::cout << "At the End of the SW algorithm" << std::endl;
}

SimpleSW::~SimpleSW(){

}

// Implementation 
__global__ void DP(char* dT_seq, char* dTrace, int* dScore){
	// ThreadId = ptn point
	int id = threadIdx.x;
	// The acid in this thread
	char p = gcP_seq[id];
	// p-1 row line's value
	__shared__ int Hp_1[1024];
	__shared__ int Ep_1[1024];	
	// Temporary
	int Hp_1_buf = 0;
	int Ep_1_buf = 0;
	// t-1 element value
	int Ht_1 = 0;
	int Ft_1 = 0;
	// p-1 t-1 element value
	int Ht_1p_1 = 0;
	// Initialize
	Hp_1[id] = 0;
	Ep_1[id] = 0;
	// Similar score
	int sim = 0;
	int point = id * gcT_size - id;
	// Culcurate elements
	for(int t = -id; t < gcT_size; ++t){
		// Control culcurate order
		if(t<0){}
		// Get similar score
		else{
			// Compare acids
			if(dT_seq[t] == p){sim = gcMatch;}
			else{sim = gcMiss;}
		}
		// SW algorithm
		// Culcurate each elements
		Ht_1p_1 += sim;	// Diagonal
		Ht_1 += gcBegin;	// Horizon (Start)
		Ft_1 += gcExtend;	// Horizon (Extend)
		Hp_1_buf = Hp_1[id] + gcBegin;	// Vertical (Start)
		Ep_1_buf = Ep_1[id] + gcExtend;	// Vertical (Extend)
		// Choose the gap score
		if(Ht_1 > Ft_1){Ft_1 = Ht_1;}	// Horizon
		if(Hp_1_buf > Ft_1){Ep_1_buf = Hp_1_buf;}	// Vertical
		// Choose the max score
		// Ht_1 is stored the max score
		if(Ht_1p_1 > Ep_1_buf){
			// Diagonal
			if(Ht_1p_1 > Ft_1){
				Ht_1 = Ht_1p_1;
				dTrace[point] = Diagonal;
			}
			// Horizon
			else{
				Ht_1 = Ft_1;
				dTrace[point] = Horizon;
			}
		}
		else {
			// Vertical
			if(Ep_1_buf > Ft_1){
				Ht_1 = Ep_1_buf;
				dTrace[point] = Vertical;
			}
			// Horizon
			else{
				Ht_1 = Ft_1;
				dTrace[point] = Horizon;
			}
		}
		// The case 0 is max
		if(Ht_1 <= 0){
			Ht_1 = 0;
			// Set 0 other value
			Ft_1 = 0;
			Ep_1_buf = 0;
			dTrace[point] = Zero;
		}
		// Hp-1 is next Ht-1p-1 
		Ht_1p_1 = Hp_1[id];
		__syncthreads();
		// Set value need next culcurate
		// p+1 row line
		if(t >= 0){
			Hp_1[id + 1] = Ht_1;
			Ep_1[id + 1] = Ep_1_buf;
			// DEBUG, score check
			// dTrace[point] = (char)(Ht_1);
		}
		if(Ht_1 >= gcThre){
	//		printf("Score = %d:\n", Ht_1);
			// traceback(dTrace, dT_seq, point-1, t);
			if(Ht_1 >= (dScore[t] & 0x0000ffff)){
			// Set score and now ptn point
				dScore[t] = Ht_1 + (id << 16);
			}
		} 
		++point;
		__syncthreads();
		// for end
	}
}

// Provisional
void SimpleSW::call_DP(const Data& txt, const Data& ptn){
	// Set txt
	char* dT_seq;
	cudaMalloc((void**)&dT_seq, sizeof(char)*txt.size());
	cudaMemcpy(dT_seq, txt.data(), sizeof(char)*txt.size(), cudaMemcpyHostToDevice);
	// Set Traceback
	char* dTrace;
	cudaMalloc((void**)&dTrace, sizeof(char)*txt.size()*ptn.size());
	// Set Score and point
	int* dScore;
	cudaMalloc((void**)&dScore, sizeof(int)*txt.size());
	int* init0 = new int[txt.size()];
	for(int i=0;i<txt.size();++i){init0[i]=0;}
	cudaMemcpy(dScore, init0, sizeof(int)*txt.size(), cudaMemcpyHostToDevice);
	// Main process
	DP<<<1,ptn.size()>>>(dT_seq, dTrace, dScore);	
	// Direction copy
	char* direction = new char[txt.size()*ptn.size()];
	cudaMemcpy(direction, dTrace, sizeof(char)*txt.size()*ptn.size(), cudaMemcpyDeviceToHost);	
	// Degug 
//	show(direction,txt,ptn);
	// Score and point copy
	int* score = new int[txt.size()];
	cudaMemcpy(score, dScore, sizeof(int)*txt.size(), cudaMemcpyDeviceToHost);
	// traceback if txt has homelogy
	checkScore(direction, score, txt);
	delete[] direction;
	delete[] score;
	delete[] init0;
	cudaFree(dT_seq);
	cudaFree(dTrace);
	cudaFree(dScore);
}

// score -> 0~16 : 17~31 = score : point of ptn
void SimpleSW::checkScore(const char* direction, const int* score, const Data& txt) const{
	// get the max score
	int x = 0, y = 0, max = 0;
	for(int i=0; i<txt.size(); ++i){
		int result = score[i] & 0x0000ffff;
		if(max < result){
			x = i;	
			y = (score[i] & 0xffff0000) >> 16;	
			max = result;
		}
	}
	std::cout << "max score is " << max << std::endl;
	if(max != 0){
		traceback(direction, txt, x, y);
	}
}

void SimpleSW::traceback(const char* direction, const Data& txt, int txt_point, int ptn_point) const{
	// Store the result, get enough size
	char *ans = new char[1024 * 2];
	// Point of result array
	int p = 0;
	int point = txt_point;
	int trace = point + ptn_point * txt.size();
	// Traceback
	while(trace >= 0){
		switch(direction[trace]){
		case Diagonal:
			ans[p++] = txt[point--];
			trace -= txt.size() + 1;
			break;
		case Vertical:
			ans[p++] = '+';
			trace -= txt.size();
			break;
		case Horizon:
			ans[p++] = '-';
			--trace;
			--point;
			break;
		case Zero:	// End
			trace = -1;
		}
	}
	// This array has reverse answer
	for(int i=p-1;i>=0;--i){ printf("%c", ans[i]); }
	printf("  %d ~ %d \n", point+1, txt_point);
	delete[] ans;
}


void SimpleSW::show(const char* score, const Data& txt, const Data& ptn) const{
	std::cout << "  ";
	for(int i=0; i < ptn.size(); ++i){
		std::cout << "  "  << ptn[i];
	}
	std::cout << std::endl;
	for(int t=0; t < txt.size(); ++t){
		std::cout << txt[t] << " ";
		for(int p=0; p < ptn.size(); ++p){
			std::cout << std::setw(3) << static_cast<int>(score[t + p*txt.size()]);
		}
		std::cout << std::endl;
	}
}

