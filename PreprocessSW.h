#ifndef PREPROCESSSW_CUH
#define PREPROCESSSW_CUH

class Data;
class Preprocess;

class PreprocessSW{
public:
	PreprocessSW(const Data& txt, const Data& ptn, const Preprocess& prepro, int threshold = 0xffff);
	~PreprocessSW();
private:
	void call_DP(const Data& txt, const Data& ptn, const Preprocess& prepro);
	void checkScore(const char* direction, const int* score, const Data& txt) const;
	void traceback(const char* direction, const Data& txt, int txt_point, int ptn_point) const; 
	void show(const char* score, const Data& txt, const Data& ptn) const;
};


#endif





