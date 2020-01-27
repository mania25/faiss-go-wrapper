//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissProductClusteringDB* newFaissProductClusteringDB(int dimension, char *faissIndexType) {
    return new FaissProductClusteringDB(dimension, faissIndexType);
}

void ReadFaissDBFromFile(FaissProductClusteringDB* fdb, char fileName[]) {
    fdb->ReadFaissDBFromFile(fileName);
}

void InitFaissDB(FaissProductClusteringDB* fdb) {
    fdb->InitFaissDB();
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

void AddNewVectorWithIDs(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[], long long pids[]) {
    fdb->AddNewVectorWithIDs(sizeOfDatabase, vectors, pids);
}

void SearchVector(FaissProductClusteringDB* fdb, int numOfQuery, float vectors[], int kTotal, float distances[], long long pids[]) {
    fdb->SearchVector(numOfQuery, vectors, kTotal, distances, pids);
}

void SearchVectorByID(FaissProductClusteringDB* fdb, long long pid, float vectors[]) {
    fdb->SearchVectorByID(pid, vectors);
}

void DeleteVectorsByIDs(FaissProductClusteringDB* fdb, int pids[]) {
    fdb->DeleteVectorsByIDs(pids);
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