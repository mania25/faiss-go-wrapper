//
// Created by Abdurrahman on 07/01/20.
//

#include <faiss/Clustering.h>
#include "faiss/IndexFlat.h"

#ifndef _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
#define _FAISS_PRODUCT_CLUSTERING_FAISS_HPP

class cxxFaissProductClusteringDB {
private:
    int dimension;
    char *faissIndexType;
    faiss::Index *faissIndex;
    std::vector<float *> listOfTrainVectors;
public:
    cxxFaissProductClusteringDB(int dimension, char *faissIndexType);

    void ReadFaissDBFromFile(char fileName[]);

    void InitFaissDB();

    void BuildIndex();

    void PushTrainDataVector(float vectors[]);

    int GetTrainDataSize();

    void AddNewVector(int sizeOfDatabase, int pids[], float vectorsFloat[]);

    int GetVectorTotal();

    void DumpFaissDB(char fileName[]);
};

#endif // _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
