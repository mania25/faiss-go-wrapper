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

int GetTrainDataSize(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    return faissProductClusteringDb->GetTrainDataSize();
}

void AddNewVector(FaissProductClusteringDB fdb, int sizeOfDatabase, int pids[], float vectorsFloat[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->AddNewVector(sizeOfDatabase, pids, vectorsFloat);
}

int GetVectorTotal(FaissProductClusteringDB fdb) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    return faissProductClusteringDb->GetVectorTotal();
}

void DumpFaissDB(FaissProductClusteringDB fdb, char fileName[]) {
    auto *faissProductClusteringDb = (cxxFaissProductClusteringDB *) fdb;
    faissProductClusteringDb->DumpFaissDB(fileName);
}