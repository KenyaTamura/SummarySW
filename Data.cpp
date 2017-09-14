#include"Data.h"
#include<fstream>
#include<iostream>
#include<random>

using namespace std ;

Data::Data(const char* fname){
	ifstream ifs(fname);
	if (ifs.fail()) {
		cerr << "Not found file" << endl;
	}	
	else{
		cout << "Loading " << fname << endl; 
	}
	ifs.seekg(0, ifstream::end);
	mSize = static_cast<int>(ifs.tellg());
	ifs.seekg(0, ifstream::beg);
	mData = new char[mSize];
	ifs.read(mData, mSize);
	if(mData[mSize-1]=='\0' ||
		mData[mSize-1]=='\n'){--mSize;}
}

Data::Data(int num) : mSize{ num } {
	mData = new char[num + 1];
	random_device rnd;
	for (int i = 0; i < num; ++i) {
		Acid acid = static_cast<Acid>(rnd() % 4);
		switch (acid){
		case Acid::A :
			mData[i] = 'A'; break;
		case Acid::G :
			mData[i] = 'G'; break;
		case Acid::C :
			mData[i] = 'C'; break;
		case Acid::T :
			mData[i] = 'T'; break;
		default:
			break;
		}
	}
	mData[num] = '\0';
//	cout << "Making data \'" << mData << "\'" << endl;
}

Data::~Data(){
	delete[] mData;
}

char Data::operator[](int i) const {
	if(i < mSize){
		return mData[i];
	}
	else{
		cerr << "Out of range" << endl;
	}
}

int Data::size() const {
	return mSize;
}

const char* Data::data() const{
	return mData;
}

