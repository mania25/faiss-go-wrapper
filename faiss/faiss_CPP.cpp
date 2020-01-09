//
// Created by Abdurrahman on 07/01/20.
//

#include "faiss_CPP.h"
#include "faiss/index_io.h"
#include <iostream>

cxxFaissProductClusteringDB::cxxFaissProductClusteringDB(int dimension, int nClusters) {
    this->dimension = dimension;
    this->nClusters = nClusters;
}

void cxxFaissProductClusteringDB::InitFaissDB() {
    indexFlatL2 = new faiss::IndexFlatL2(this->dimension);
    indexIVFFlat = new faiss::IndexIVFFlat(indexFlatL2, this->dimension, this->nClusters, faiss::METRIC_L2);
    indexIVFFlat->verbose = true;
}

void cxxFaissProductClusteringDB::BuildIndex() {
    indexIVFFlat->train(listOfTrainVectors.size(), *listOfTrainVectors.data());
}

void cxxFaissProductClusteringDB::PushTrainDataVector(float *vectors) {
    listOfTrainVectors.push_back(vectors);
}

int cxxFaissProductClusteringDB::GetTrainDataSize() {
    return listOfTrainVectors.size();
}

void cxxFaissProductClusteringDB::AddNewVector(int sizeOfDatabase, int pids[], float vectors[]) {
    indexIVFFlat->add_with_ids(sizeOfDatabase, vectors, (int64_t*)pids);
}

int cxxFaissProductClusteringDB::GetVectorTotal() {
    return indexIVFFlat->ntotal;
}

void cxxFaissProductClusteringDB::DumpFaissDB(char fileName[]) {
    faiss::write_index(indexFlatL2, fileName);
}