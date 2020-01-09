//
// Created by Abdurrahman on 07/01/20.
//

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

#ifndef _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
#define _FAISS_PRODUCT_CLUSTERING_FAISS_HPP

class cxxFaissProductClusteringDB {
private:
    faiss::IndexFlatL2* indexFlatL2{};
    faiss::IndexIVFFlat* indexIVFFlat{};
    int dimension;
    int nClusters;
    std::vector<float*> listOfTrainVectors;
public:
    cxxFaissProductClusteringDB(int dimension, int nClusters);
    void InitFaissDB();
    void BuildIndex();
    void PushTrainDataVector(float vectors[]);
    int GetTrainDataSize();
    void AddNewVector(int sizeOfDatabase, int pids[], float vectorsFloat[]);
    int GetVectorTotal();
};

#endif // _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
