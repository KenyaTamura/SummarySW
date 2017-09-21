#include<string>
#include<iostream>

#include"Measure.h"
#include"Writing.h"
#include"Timer.h"

#include"Data.h"
#include"SimpleSW.h"
#include"Preprocess.h"
#include"PreprocessSW.h"
#include"SeparateSW.h"
#include"PostponeSW.h"

using namespace std;

Measure::Measure(const char* fname){
	mTxt = new Data(fname);
	mWSimple = new Writing("result/RSimple.csv");
	mWPre = new Writing("result/RPreprocess.csv");
	mWPost = new Writing("result/RPostpone.csv");
	mWSepa = new Writing("result/RSeparate.csv");
}

Measure::~Measure(){
	delete mTxt;
	delete mWSimple;
	delete mWPre;
	delete mWPost;
	delete mWSepa;
}

void Measure::exp(int ptn_size, int threshold, int times){
	Timer t;
	string simple, pre, separate, post;
	auto in = [](int x){ return to_string(x) + ','; };
	simple = 'p' + to_string(ptn_size) + "thre" + to_string(threshold) + ',';
	pre = simple;
	separate = simple;
	post = simple;
	// First time has some loss time
	PostponeSW(*mTxt, Data(ptn_size), threshold);
	for(int i=0; i<times;++i){
		Data ptn(ptn_size);
		t.start();	SimpleSW(*mTxt, ptn, threshold);	
		simple += in(t.finish());
		t.start();	PreprocessSW(*mTxt, ptn, Preprocess(*mTxt, ptn, threshold), threshold);
		pre += in(t.finish());
		t.start();	SeparateSW(*mTxt, ptn, 8, threshold);
		separate += in(t.finish());
		t.start(); PostponeSW(*mTxt, ptn, threshold);
		post += in(t.finish());
	}
	mWSimple->write(simple.c_str());
	mWPre->write(pre.c_str());
	mWSepa->write(separate.c_str());
	mWPost->write(post.c_str());
}
