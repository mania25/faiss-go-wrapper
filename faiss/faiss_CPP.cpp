//
// Created by Abdurrahman on 07/01/20.
//

#include "faiss_CPP.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/impl/AuxIndexStructures.h"

cxxFaissProductClusteringDB::cxxFaissProductClusteringDB(int dimension, char *faissIndexType) {
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}

void cxxFaissProductClusteringDB::ReadFaissDBFromFile(char *fileName) {
    faissIndex = faiss::read_index(fileName);
    faissIndex->verbose = true;
}

void cxxFaissProductClusteringDB::InitFaissDB() {
    faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, faiss::METRIC_L2);
    faissIndex->verbose = true;
}

void cxxFaissProductClusteringDB::BuildIndex() {
    faissIndex->train(listOfTrainVectors.size(), *listOfTrainVectors.data());
}

void cxxFaissProductClusteringDB::PushTrainDataVector(float *vectors) {
    listOfTrainVectors.push_back(vectors);
}

int cxxFaissProductClusteringDB::GetTrainDataSize() {
    return listOfTrainVectors.size();
}

void cxxFaissProductClusteringDB::AddNewVector(int sizeOfDatabase, int pids[], float vectors[]) {
    faissIndex->add_with_ids(sizeOfDatabase, vectors, (int64_t *) pids);
}

void
cxxFaissProductClusteringDB::SearchVector(int numOfQuery, float *vectors, int kTotal, float distances[], int64_t pids[]) {
    faissIndex->search(numOfQuery, vectors, kTotal, distances,
                       pids);
}

void cxxFaissProductClusteringDB::SearchVectorByID(int64_t pid, float vectors[]) {
    faissIndex->reconstruct(pid, vectors);
}

void cxxFaissProductClusteringDB::DeleteVectorsByIDs(int pids[]) {
    auto pidsLen = *(&pids + 1) - pids;
    faissIndex->remove_ids(faiss::IDSelectorBatch(pidsLen, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
}

int cxxFaissProductClusteringDB::GetVectorTotal() {
    return faissIndex->ntotal;
}

void cxxFaissProductClusteringDB::DumpFaissDB(char fileName[]) {
    faiss::write_index(faissIndex, fileName);
}

void cxxFaissProductClusteringDB::ResetIndex() {
    faissIndex->reset();
}


