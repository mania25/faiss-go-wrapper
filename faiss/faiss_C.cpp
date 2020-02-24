//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissProductClusteringDB* newFaissProductClusteringDB(int dimension, char *faissIndexType) {
    return new FaissProductClusteringDB(dimension, faissIndexType);
}

void ReadFaissDBFromFile(FaissProductClusteringDB* fdb, char fileName[], int ioflags) {
    fdb->ReadFaissDBFromFile(fileName, ioflags);
}

void InitFaissDB(FaissProductClusteringDB* fdb, int metricType) {
    fdb->InitFaissDB(metricType);
}

void PushTrainDataVector(FaissProductClusteringDB* fdb, float vectors[]) {
    fdb->PushTrainDataVector(vectors);
}

void ValidateTrainDataset(FaissProductClusteringDB* fdb) {
    fdb->ValidateTrainDataset();
}

unsigned long GetTrainDataSize(FaissProductClusteringDB* fdb) {
    return fdb->GetTrainDataSize();
}

void BuildIndex(FaissProductClusteringDB* fdb, int numOfTrainDataset) {
    fdb->BuildIndex(numOfTrainDataset);
}

bool GetTrainStatus(FaissProductClusteringDB* fdb) {
    return fdb->GetTrainStatus();
}

void AddNewVector(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[]) {
    fdb->AddNewVector(sizeOfDatabase, vectors);
}

void AddNewVectorWithIDs(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[], int64_t pids[]) {
    fdb->AddNewVectorWithIDs(sizeOfDatabase, vectors, pids);
}

void SearchVector(FaissProductClusteringDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]) {
    fdb->SearchVector(numOfQuery, nProbe, vectors, kTotal, distances, pids);
}

void SearchVectorByID(FaissProductClusteringDB* fdb, int64_t pid, int nProbe, float vectors[]) {
    fdb->SearchVectorByID(pid, nProbe, vectors);
}

void SearchCentroidIDByVector(FaissProductClusteringDB *fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
    fdb->SearchCentroidIDByVector(vectors, numOfQuery, nProbe, clusterIDs);
}

void DeleteVectorsByIDs(FaissProductClusteringDB* fdb, size_t numOfQuery, int pids[]) {
    fdb->DeleteVectorsByIDs(numOfQuery, pids);
}

int GetVectorTotal(FaissProductClusteringDB* fdb) {
    return fdb->GetVectorTotal();
}

void DumpFaissDB(FaissProductClusteringDB* fdb, char fileName[]) {
    fdb->DumpFaissDB(fileName);
}

void ResetIndex(FaissProductClusteringDB* fdb) {
    fdb->ResetIndex();
}
