#ifndef WRITING_H
#define WRITING_H

#include<fstream>

class Writing{
public:
	Writing(const char* fname);
	~Writing();
	void write(const char* data);
private:
	std::ofstream OFile;
};

#endif

