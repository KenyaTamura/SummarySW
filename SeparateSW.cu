#include<cuda.h>
#include<nvvm.h>
#include<unistd.h>
#include<malloc.h>

#include<iostream>
#include<iomanip>

#include"SeparateSW.h"
#include"Data.h"
#include"Gain.h"

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

SeparateSW::SeparateSW(const Data& txt, const Data& ptn, int block_num, int threshold) : mBlock{ block_num }
{
	cout << "At the beginning of the SW algorithm (Separate)" << endl; 
	// Sieze check
	if(ptn.size() > 1024){
		cout << "Too large size" << endl;
		return;
	}
	cout << "Block number = " << mBlock << endl;
	// Set value in constant memory
	int tsize = txt.size();
	int psize = ptn.size();
	cudaMemcpyToSymbol(gcT_size, &tsize, sizeof(int));
	cudaMemcpyToSymbol(gcP_size, &psize, sizeof(int));
	cudaMemcpyToSymbol(gcThre, &threshold, sizeof(int));
	cudaMemcpyToSymbol(gcP_seq, ptn.data(), sizeof(char) * ptn.size());
	// Cost and gain
	int gain = MATCH;
	cudaMemcpyToSymbol(gcMatch, &gain, sizeof(int));
	gain = MISS;
	cudaMemcpyToSymbol(gcMiss, &gain, sizeof(int));
	gain = EXT;
	cudaMemcpyToSymbol(gcExtend, &gain, sizeof(int));
	gain = BEG;
	cudaMemcpyToSymbol(gcBegin, &gain, sizeof(int));
	// Dynamic Programing part by call_DP
	call_DP(txt, ptn);
	cout << "At the end of the SW algorithm" << endl;
}

SeparateSW::~SeparateSW(){

}

// Implementation 
__global__ void DP(char* dT_seq, char* dTrace, int* dScore, int block_size){
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
	// Set start point
	int start = (block_size - gcP_size) * blockIdx.x;
	// Score point
	int score_point = block_size * blockIdx.x;
	if(blockIdx.x != 0){ start -= gcP_size; }	// Take margin other than 0 block 
	// The traceback point
	int point = ((gcP_size * block_size) * blockIdx.x) 
				+ (id * block_size) 
				- id;
	// Culcurate elements
	for(int t = start - id; t < start + block_size; ++t){
		// Control culcurate order
		if(t<0 || t < start){}
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
		// The case 0 is max && in search range
		if(Ht_1 <= 0 && sim != 0){
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
		if(t >= 0 && sim != 0){
			Hp_1[id + 1] = Ht_1;
			Ep_1[id + 1] = Ep_1_buf;
			// DEBUG, score check
	//		dTrace[point] = (char)(Ht_1);
		}
		if(Ht_1 >= gcThre){
			// printf("Score = %d:\n", Ht_1);
			// traceback(dTrace, dT_seq, point-1, t);
			if(Ht_1 >= (dScore[score_point + t - start] & 0x0000ffff)){
			// Set score and now ptn point
				dScore[score_point + t - start] = Ht_1 + (id << 16);
			}
		} 
		++point;
		__syncthreads();
		// for end
	}
}

// Provisional
void SeparateSW::call_DP(const Data& txt, const Data& ptn){
	// Set txt
	char* dT_seq;
	cudaMalloc((void**)&dT_seq, sizeof(char)*txt.size());
	cudaMemcpy(dT_seq, txt.data(), sizeof(char)*txt.size(), cudaMemcpyHostToDevice);
	// Set block size
	int block_size = (txt.size()/mBlock) + ptn.size();
	// Set Traceback
	char* dTrace;
	cudaMalloc((void**)&dTrace, sizeof(char) * block_size * mBlock * ptn.size());
	// Set Score and point
	int* dScore;
	cudaMalloc((void**)&dScore, sizeof(int) * block_size * mBlock);
	int* init0 = new int[block_size * mBlock];
	for(int i=0;i<block_size * mBlock;++i){init0[i]=0;}
	cudaMemcpy(dScore, init0, sizeof(int) * block_size * mBlock, cudaMemcpyHostToDevice);
	// Main process
	DP<<<mBlock,ptn.size()>>>(dT_seq, dTrace, dScore, block_size);	
	// Score or direction check
	char* direction = new char[block_size * mBlock * ptn.size()];
	cudaMemcpy(direction, dTrace, sizeof(char) * block_size * mBlock * ptn.size(), cudaMemcpyDeviceToHost);	
	int* score = new int[block_size * mBlock];
	cudaMemcpy(score, dScore, sizeof(int) * block_size * mBlock, cudaMemcpyDeviceToHost);
//	show(direction, txt, ptn);
	checkScore(direction, score, txt, ptn);
	delete[] score;
	delete[] direction;
	delete[] init0;
	cudaFree(dT_seq);
	cudaFree(dTrace);
	cudaFree(dScore);
}

// score -> 0~16 : 17~31 = score : point of ptn
void SeparateSW::checkScore(const char* direction, const int* score, const Data& txt, const Data& ptn) const{
	// get the max score
	int x = 0, y = 0, max = 0, block = 0;
	int block_size = (txt.size() / mBlock) + ptn.size();
	for(int b=0; b<mBlock; ++b){
		int offset = b*block_size;
		for(int i=0; i<block_size; ++i){
			int result = score[offset + i] & 0x0000ffff;
			if(max < result){	
				y = (score[offset + i] & 0xffff0000) >> 16;	
				block = b;
				x = i;
				max = result;
			}
		}
	}
	cout << "max score is " << max << endl;
	if(max != 0){
		traceback(direction, txt, ptn, x, y, block);
	}
}

void SeparateSW::traceback(const char* direction, const Data& txt, const Data& ptn, int x, int y, int block) const{
	// Store the result, get enough size
	char *ans = new char[ptn.size() * 2];
	// Point of result array
	int p = 0;
	int block_size = (txt.size() / mBlock) + ptn.size();
	int offset = block_size * block * ptn.size();
	// Point in the block
	int trace = x + y * block_size;
	// Real txt point
	int txt_point = ((txt.size() / mBlock) * block) + x;
	txt_point -= (block==0) ? 0 : ptn.size();
	int end_point = txt_point;	
	// Traceback
	while(trace >= 0){
		switch(direction[offset + trace]){
		case Diagonal:
			ans[p++] = txt[txt_point--];
			trace -= block_size + 1;
			break;
		case Vertical:
			ans[p++] = '+';
			trace -= block_size;
			break;
		case Horizon:
			ans[p++] = '-';
			--trace;
			--txt_point;
			break;
		case Zero:	// End
			trace = -1;
			break;
		default:	// Didn't use
			trace = -1;
			break;
		}
	}
	// This array has reverse answer
	for(int i=p-1;i>=0;--i){ cout << ans[i]; }
	printf("  %d ~ %d \n", txt_point, end_point);
	delete[] ans;
}



void SeparateSW::show(const char* direction, const Data& txt, const Data& ptn) const{
	int block_size = txt.size() / mBlock + ptn.size();	
	for(int k=0; k < mBlock; ++k){
		cout << "\n  ";
		for(int i=0; i < ptn.size(); ++i){
			cout << "  "  << ptn[i];
		}
		cout << endl;
		int point = (block_size - ptn.size()) * k;
		point -= (point==0) ? 0 : ptn.size();
		for(int t=0; t < block_size; ++t){
			cout << txt[point + t] << " ";
			for(int p=0; p < ptn.size(); ++p){
				cout << setw(3) 
					<< static_cast<int>(direction[(block_size*ptn.size()*k) + (t + p*block_size)]);
			}
			cout << endl;
		}
	}
}

