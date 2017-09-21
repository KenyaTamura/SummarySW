#ifndef MEASURE_H
#define MEASURE_H

class Writing;
class Data;
// Set condition of experiment and times
// Write result in each method's memo
class Measure{
public:
	Measure(const char* fname);
	~Measure();
	void exp(int ptn_size, int threshold, int times);
private:
	Data* mTxt;
	Writing *mWSimple, *mWPre, *mWPost, *mWSepa;
};


#endif
