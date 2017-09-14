#ifndef KASE_H
#define KASE_H

#include"Data.h"
#include<list>

class Preprocess {
public:
	Preprocess(const Data& txt, const Data& ptn, const int Threshold = 0xffff);
	~Preprocess();
	int get(int i) const;
	const int* getAll() const;
	int block() const;
private:
	// 0,13,25,50 = 0~13 and 25~50
	int* mRange;
	// The number of range blocks
	int mBlock;

	// The number of each acid
	struct Hash {
		int ADE = 0;
		int GUA = 0;
		int CYT = 0;
		int TYM = 0;
	};
	// Hash plus
	inline void plus_hash(Hash& h, char acid) const;
	// Hash minus
	inline void minus_hash(Hash& h, char acid) const;
	// Main process
	/// txt = long sequence, ptn = short sequence, threshold = Border of OK, range = set result 
	void get_range(const Data& txt, const Data& ptn, const int threshold, std::list<int>& range);
	// Get hash
	Hash get_hash(const Data& data, int length) const;
	// Compare hash
	int get_score(const Hash& hash1, const Hash& hash2) const;
	// Shaping
	void shape(std::list<int>& origin, const int interval);
};


#endif

