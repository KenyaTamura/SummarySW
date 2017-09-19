#ifndef WRITER_H
#define WRITER_H

#include<fstream>

class Writer{
public:
	Writer(const char* fname);
	~Writer();
	void write(const char* data);
private:
	std::ostream OFile;
};

#endif

