//
// Created by Abdurrahman on 07/01/20.
//

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

    void SearchVector(int numOfQuery, float vectors[], int kTotal, float distances[], int64_t pids[]);

    void SearchVectorByID(int64_t pid, float vectors[]);

    void DeleteVectorsByIDs(int pids[]);

    int GetVectorTotal();

    void DumpFaissDB(char fileName[]);

    void ResetIndex();
};

#endif // _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
