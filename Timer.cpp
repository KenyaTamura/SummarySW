#include"Timer.h"

Timer::Timer(){
}

Timer::~Timer(){
}

void Timer::start(){
	mStart = std::chrono::system_clock::now();
}

int Timer::finish(){
	mEnd = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

