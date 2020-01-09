//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissProductClusteringDB getFaissProductClusteringDB( int dimension, int nClusters ){
    cxxFaissProductClusteringDB *faissProductClusteringDb = new cxxFaissProductClusteringDB(dimension, nClusters);
    return (void*)faissProductClusteringDb;
}

void ReadFaissDBFromFile(FaissProductClusteringDB fdb, char fileName[]) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->ReadFaissDBFromFile(fileName);
}

void InitFaissDB(FaissProductClusteringDB fdb) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->InitFaissDB();
}

void BuildIndex(FaissProductClusteringDB fdb) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->BuildIndex();
}

void PushTrainDataVector(FaissProductClusteringDB fdb, float vectors[]) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->PushTrainDataVector(vectors);
}

int GetTrainDataSize(FaissProductClusteringDB fdb) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    return faissProductClusteringDb->GetTrainDataSize();
}

void AddNewVector(FaissProductClusteringDB fdb, int sizeOfDatabase, int pids[], float vectorsFloat[]) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->AddNewVector(sizeOfDatabase, pids, vectorsFloat);
}

int GetVectorTotal(FaissProductClusteringDB fdb) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    return faissProductClusteringDb->GetVectorTotal();
}

void DumpFaissDB(FaissProductClusteringDB fdb, char fileName[]) {
    cxxFaissProductClusteringDB *faissProductClusteringDb = (cxxFaissProductClusteringDB*)fdb;
    faissProductClusteringDb->DumpFaissDB(fileName);
}