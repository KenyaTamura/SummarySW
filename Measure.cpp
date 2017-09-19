#include<string>

#include"Writer.h"
#include"Timer.h"

#include"Data.h"
#include"SimpleSW.h"
#include"Preprocess.h"
#include"PreprocessSW.h"
#include"SeparateSW.h"
#include"PostponeSW.h"

Measure::Measure(const char* fname){
	mTxt = new Data(fname);
}

Measure::~Measure(){
	delete mTxt;
}

void Measure::exp(int ptn_size, int threshold, int times) const{
	Timer t;
	string simple, pre, separate, post;
	auto in = [](int x){ return to_string + ','; };
	auto time_in = [](){ return in(t.finish()); };
	simple = in(ptn_size) + to_string(threshold);
	pre = simple;
	separate = simple;
	post = simple;
	for(int i=0; i<times;++i){
		Data ptn(20);
		t.start();	SimpleSW(mTxt, ptn, threshold);	
		simple += time_in;
		t.start();	PreprocessSW(mTxt, ptn, Preprocess(mTxt, ptn, threshold), threshold);
		pre += time_in;
		t.start();	SeparateSW(mTxt, ptn, 8, threshold);
		separate += time_in;
		t.start(); PostponeSW(mTxt, ptn, threshold);
		post += time_in;
	}
	// TODO write file
}
