//
// Created by Abdurrahman on 07/01/20.
//

#include "faiss/IndexFlat.h"

#ifndef _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
#define _FAISS_PRODUCT_CLUSTERING_FAISS_HPP

class FaissProductClusteringDB {
private:
    int dimension;
    const char *faissIndexType;
    faiss::Index *faissIndex;
    std::vector<float> listOfTrainVectors;
public:
    FaissProductClusteringDB(int dimension, const char *faissIndexType);

    void ReadFaissDBFromFile(char fileName[]);

    void InitFaissDB();

    void PushTrainDataVector(const float vectors[]);

    void ValidateTrainDataset();

    void BuildIndex(int numOfTrainDataset);

    bool GetTrainStatus();

    u_long GetTrainDataSize();

    void AddNewVectorWithIDs(int sizeOfDatabase, float vectors[], long long pids[]);

    void SearchVector(int numOfQuery, float vectors[], int kTotal, float distances[], long long pids[]);

    void SearchVectorByID(long long pid, float vectors[]);

    void DeleteVectorsByIDs(int pids[]);

    int GetVectorTotal();

    void DumpFaissDB(const char fileName[]);

    void ResetIndex();
};

#endif // _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
