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

    void ReadFaissDBFromFile(char fileName[], int ioflags);

    void InitFaissDB();

    void PushTrainDataVector(const float vectors[]);

    void ValidateTrainDataset();

    u_long GetTrainDataSize();

    void BuildIndex(int numOfTrainDataset);

    bool GetTrainStatus();

    void AddNewVector(int sizeOfDatabase, float vectors[]);

    void AddNewVectorWithIDs(int sizeOfDatabase, float vectors[], int64_t pids[]);

    void SearchVector(int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]);

    void SearchVectorByID(int64_t pid, int nProbe, float vectors[]);

    void SearchCentroidIDByVector(float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs);

    void DeleteVectorsByIDs(size_t numOfQuery, int pids[]);

    int GetVectorTotal();

    void DumpFaissDB(const char fileName[]);

    void ResetIndex();
};

#endif // _FAISS_PRODUCT_CLUSTERING_FAISS_HPP
