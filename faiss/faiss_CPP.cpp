//
// Created by Abdurrahman on 07/01/20.
//

#include <math.h>
#include "faiss_CPP.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/impl/AuxIndexStructures.h"

cxxFaissProductClusteringDB::cxxFaissProductClusteringDB(int dimension, char *faissIndexType) {
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}

void cxxFaissProductClusteringDB::ReadFaissDBFromFile(char *fileName) {
    this->faissIndex = faiss::read_index(fileName);
    this->faissIndex->verbose = true;
}

void cxxFaissProductClusteringDB::InitFaissDB() {
    this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, faiss::METRIC_L2);
    this->faissIndex->verbose = true;
}

void cxxFaissProductClusteringDB::BuildIndex() {
    this->faissIndex->train(this->listOfTrainVectors.size(), this->listOfTrainVectors.data());
}

void cxxFaissProductClusteringDB::PushTrainDataVector(float *vectors) {
    this->listOfTrainVectors.push_back(*vectors);
}

void cxxFaissProductClusteringDB::ValidateTrainDataset() {
    for (int i = 0; i < listOfTrainVectors.size(); ++i) {
        float *data = &listOfTrainVectors[i];

        for (int j = 0; j < listOfTrainVectors.size() * dimension; ++j) {
            if (!std::isfinite(data[j])){
                printf("Invalid vectors data, Got: %f", data[j]);
                return;
            }
        }
    }
}

int cxxFaissProductClusteringDB::GetTrainDataSize() {
    return this->listOfTrainVectors.size();
}

void cxxFaissProductClusteringDB::AddNewVector(int sizeOfDatabase, int pids[], float vectors[]) {
    this->faissIndex->add_with_ids(sizeOfDatabase, vectors, (int64_t *) pids);
}

void
cxxFaissProductClusteringDB::SearchVector(int numOfQuery, float *vectors, int kTotal, float distances[], int64_t pids[]) {
    this->faissIndex->search(numOfQuery, vectors, kTotal, distances,
                       pids);
}

void cxxFaissProductClusteringDB::SearchVectorByID(int64_t pid, float vectors[]) {
    this->faissIndex->reconstruct(pid, vectors);
}

void cxxFaissProductClusteringDB::DeleteVectorsByIDs(int pids[]) {
    auto pidsLen = *(&pids + 1) - pids;
    this->faissIndex->remove_ids(faiss::IDSelectorBatch(pidsLen, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
}

int cxxFaissProductClusteringDB::GetVectorTotal() {
    return this->faissIndex->ntotal;
}

void cxxFaissProductClusteringDB::DumpFaissDB(char fileName[]) {
    faiss::write_index(this->faissIndex, fileName);
}

void cxxFaissProductClusteringDB::ResetIndex() {
    this->faissIndex->reset();
}


