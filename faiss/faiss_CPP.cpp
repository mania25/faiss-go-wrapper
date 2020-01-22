//
// Created by Abdurrahman on 07/01/20.
//

#include <math.h>
#include "faiss_CPP.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/impl/AuxIndexStructures.h"

FaissProductClusteringDB::FaissProductClusteringDB(int dimension, const char *faissIndexType) {
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}

void FaissProductClusteringDB::ReadFaissDBFromFile(char *fileName) {
    this->faissIndex = faiss::read_index(fileName);
    this->faissIndex->verbose = true;
}

void FaissProductClusteringDB::InitFaissDB() {
    this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, faiss::METRIC_L2);
    this->faissIndex->verbose = true;
}

void FaissProductClusteringDB::BuildIndex(int numOfTrainDataset) {
    this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
}

void FaissProductClusteringDB::PushTrainDataVector(const float vectors[]) {
    for (int i = 0; i < (sizeof(*vectors) / sizeof(float)) * this->dimension; ++i) {
        this->listOfTrainVectors.push_back(vectors[i]);
    }
}

void FaissProductClusteringDB::ValidateTrainDataset() {
    for (float listOfTrainVector : listOfTrainVectors) {
        float data = listOfTrainVector;

        if (!std::isfinite(data)){
            printf("Invalid vectors data, Got: %f", data);
            return;
        }
    }
}

u_long FaissProductClusteringDB::GetTrainDataSize() {
    return static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
}

void FaissProductClusteringDB::AddNewVector(int sizeOfDatabase, int pids[], float vectors[]) {
    this->faissIndex->add_with_ids(sizeOfDatabase, vectors, (int64_t *) pids);
}

void FaissProductClusteringDB::SearchVector(int numOfQuery, float *vectors, int kTotal, float distances[], int64_t pids[]) {
    this->faissIndex->search(numOfQuery, vectors, kTotal, distances,
                       pids);
}

void FaissProductClusteringDB::SearchVectorByID(int64_t pid, float vectors[]) {
    this->faissIndex->reconstruct(pid, vectors);
}

void FaissProductClusteringDB::DeleteVectorsByIDs(int pids[]) {
    auto pidsLen = *(&pids + 1) - pids;
    this->faissIndex->remove_ids(faiss::IDSelectorBatch(pidsLen, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
}

int FaissProductClusteringDB::GetVectorTotal() {
    return this->faissIndex->ntotal;
}

void FaissProductClusteringDB::DumpFaissDB(const char fileName[]) {
    faiss::write_index(this->faissIndex, fileName);
}

void FaissProductClusteringDB::ResetIndex() {
    this->faissIndex->reset();
}


