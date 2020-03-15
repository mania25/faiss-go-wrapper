//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissDB* newFaissDB(int dimension, char *faissIndexType) {
    return new FaissDB(dimension, faissIndexType);
}

void ReadFaissDBFromFile(FaissDB* fdb, char fileName[], int ioflags) {
    fdb->ReadFaissDBFromFile(fileName, ioflags);
}

void InitFaissDB(FaissDB* fdb, int metricType) {
    fdb->InitFaissDB(metricType);
}

void PreAllocateTrainVector(FaissDB* fdb, int size) {
    fdb->PreAllocateTrainVector(size);
}

void PushTrainDataVector(FaissDB* fdb, float vectors[]) {
    fdb->PushTrainDataVector(vectors);
}

void ValidateTrainDataset(FaissDB* fdb) {
    fdb->ValidateTrainDataset();
}

unsigned long GetTrainDataSize(FaissDB* fdb) {
    return fdb->GetTrainDataSize();
}

void BuildIndex(FaissDB* fdb, int numOfTrainDataset) {
    fdb->BuildIndex(numOfTrainDataset);
}

bool GetTrainStatus(FaissDB* fdb) {
    return fdb->GetTrainStatus();
}

void AddNewVector(FaissDB* fdb, int sizeOfDatabase, float vectors[]) {
    fdb->AddNewVector(sizeOfDatabase, vectors);
}

void AddNewVectorWithIDs(FaissDB* fdb, int sizeOfDatabase, float vectors[], int64_t pids[]) {
    fdb->AddNewVectorWithIDs(sizeOfDatabase, vectors, pids);
}

void SearchVector(FaissDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]) {
    fdb->SearchVector(numOfQuery, nProbe, vectors, kTotal, distances, pids);
}

void SearchVectorByID(FaissDB* fdb, int64_t pid, int nProbe, float vectors[]) {
    fdb->SearchVectorByID(pid, nProbe, vectors);
}

void SearchCentroidIDByVector(FaissDB *fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
    fdb->SearchCentroidIDByVector(vectors, numOfQuery, nProbe, clusterIDs);
}

void DeleteVectorsByIDs(FaissDB* fdb, size_t numOfQuery, int pids[]) {
    fdb->DeleteVectorsByIDs(numOfQuery, pids);
}

int GetVectorTotal(FaissDB* fdb) {
    return fdb->GetVectorTotal();
}

void DumpFaissDB(FaissDB* fdb, char fileName[]) {
    fdb->DumpFaissDB(fileName);
}

void ResetIndex(FaissDB* fdb) {
    fdb->ResetIndex();
}
