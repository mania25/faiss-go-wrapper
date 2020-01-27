//
// Created by Abdurrahman on 07/01/20.
//

#include <math.h>
#include "faiss_CPP.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/impl/AuxIndexStructures.h"
#include <faiss/IVFlib.h>

FaissProductClusteringDB::FaissProductClusteringDB(int dimension, const char *faissIndexType) {
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}

void FaissProductClusteringDB::ReadFaissDBFromFile(char *fileName) {
    this->faissIndex = faiss::read_index(fileName, faiss::IO_FLAG_MMAP);
    this->faissIndex->verbose = true;
}

void FaissProductClusteringDB::InitFaissDB() {
    this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, faiss::METRIC_L2);
    this->faissIndex->verbose = true;
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

void FaissProductClusteringDB::BuildIndex(int numOfTrainDataset) {
    this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
}

bool FaissProductClusteringDB::GetTrainStatus() {
    bool trainStatus = this->faissIndex->is_trained;
    return trainStatus;
}

u_long FaissProductClusteringDB::GetTrainDataSize() {
    return static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
}

void FaissProductClusteringDB::AddNewVectorWithIDs(int sizeOfDatabase, float vectors[], long long pids[]) {
    std::vector <float> database;
    std::vector <long long> ids;

    for (int i = 0; i < sizeOfDatabase; ++i)  {
        ids.push_back(pids[i]);
    }


    for (int i = 0; i < this->dimension; ++i) {
        database.push_back(vectors[i]);
    }

    this->faissIndex->add_with_ids(sizeOfDatabase, database.data(), ids.data());
}

void FaissProductClusteringDB::SearchVector(int numOfQuery, float *vectors, int kTotal, float *distances, long long *pids) {
    this->faissIndex->search(numOfQuery, vectors, kTotal, distances,
                       pids);
}

void FaissProductClusteringDB::SearchVectorByID(long long pid, float vectors[]) {
    faiss::ivflib::extract_index_ivf(faissIndex)->make_direct_map(true);
    faissIndex->reconstruct(pid, vectors);
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

