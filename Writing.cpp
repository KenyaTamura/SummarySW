#include<iostream>

#include"Writing.h"

Writing::Writing(const char* fname){
	OFile.open(fname, std::ios::app);	
}

Writing::~Writing(){
	OFile.close();
}

void Writing::write(const char* data){
	if(!OFile){
		std::cout << "Not open output file" << std::endl;
	}
	else{
		OFile << data << std::endl;
	}
}


