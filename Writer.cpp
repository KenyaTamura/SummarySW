#include"Writer.h"

Writer::Writer(const char* fname){
	OFile.open(fname, std::ios::app);
}

Writer::~Writer(){

}

void write(const char* data){
	OFile << data << std::endl;
}


