//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissProductClusteringDB getFaissProductClusteringDB(int dimension, char *faissIndexType) {
    auto faissProductClusteringDb = new cxxFaissProductClusteringDB(dimension, faissIndexType);
    return (void *) faissProductClusteringDb;
}

void ReadFaissDBFromFile(FaissProductClusteringDB fdb, char fileName[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->ReadFaissDBFromFile(fileName);
}

void InitFaissDB(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->InitFaissDB();
}

void BuildIndex(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->BuildIndex();
}

void PushTrainDataVector(FaissProductClusteringDB fdb, float vectors[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->PushTrainDataVector(vectors);
}

void ValidateTrainDataset(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->ValidateTrainDataset();
}

int GetTrainDataSize(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    return faissProductClusteringDb->GetTrainDataSize();
}

void AddNewVector(FaissProductClusteringDB fdb, int sizeOfDatabase, int pids[], float vectorsFloat[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->AddNewVector(sizeOfDatabase, pids, vectorsFloat);
}

void SearchVector(FaissProductClusteringDB fdb, int numOfQuery, float vectors[], int kTotal, float distances[], int64_t pids[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->SearchVector(numOfQuery, vectors, kTotal, distances, pids);
}

void SearchVectorByID(FaissProductClusteringDB fdb, int64_t pid, float vectors[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->SearchVectorByID(pid, vectors);
}

void DeleteVectorsByIDs(FaissProductClusteringDB fdb, int pids[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->DeleteVectorsByIDs(pids);
}

int GetVectorTotal(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    return faissProductClusteringDb->GetVectorTotal();
}

void DumpFaissDB(FaissProductClusteringDB fdb, char fileName[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->DumpFaissDB(fileName);
}

void ResetIndex(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->ResetIndex();
}