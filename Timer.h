#ifndef TIMER_H
#define TIMER_H

#include<chrono>

class Timer{
public:
	Timer();
	~Timer();
	void start();
	int finish(); // return time
private:
	std::chrono::system_clock::time_point mStart, mEnd;
};


#endif
