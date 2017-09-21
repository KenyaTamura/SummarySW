#include"Timer.h"

Timer::Timer(){
	mStart = std::chrono::system_clock::now();
}

Timer::~Timer(){
}

void Timer::start(){
	mStart = std::chrono::system_clock::now();
}

int Timer::finish(){
	auto end = std::chrono::system_clock::now();
	auto time = end - mStart;
	return std::chrono::duration_cast<std::chrono::microseconds>(time).count();
}

