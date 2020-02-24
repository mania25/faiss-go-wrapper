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

void FaissProductClusteringDB::ReadFaissDBFromFile(char *fileName, int ioflags) {
    try {
        this->faissIndex = faiss::read_index(fileName, ioflags);
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::InitFaissDB(int metricType) {
    try {
        this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, static_cast<faiss::MetricType>(metricType));
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
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
            printf("Invalid vectors data, Got: %f\n", data);
            return;
        }
    }
}

u_long FaissProductClusteringDB::GetTrainDataSize() {
    return static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
}

void FaissProductClusteringDB::BuildIndex(int numOfTrainDataset) {
    try {
        this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

bool FaissProductClusteringDB::GetTrainStatus() {
    bool trainStatus = this->faissIndex->is_trained;
    return trainStatus;
}

void FaissProductClusteringDB::AddNewVector(int sizeOfDatabase, float *vectors) {
    try {
        this->faissIndex->add(sizeOfDatabase, vectors);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::AddNewVectorWithIDs(int sizeOfDatabase, float vectors[], int64_t pids[]) {
    try {
        std::vector <float> database;
        std::vector <int64_t> ids;

        for (int i = 0; i < sizeOfDatabase; ++i)  {
            ids.push_back(pids[i]);
        }

        for (int i = 0; i < this->dimension; ++i) {
            database.push_back(vectors[i]);
        }

        this->faissIndex->add_with_ids(sizeOfDatabase, database.data(), ids.data());
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::SearchVector(int numOfQuery, int nProbe, float *vectors, int kTotal, float *distances, int64_t *pids) {
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        this->faissIndex->search(numOfQuery, vectors, kTotal, distances, pids);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::SearchVectorByID(int64_t pid, int nProbe, float vectors[]) {
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        this->faissIndex->reconstruct(pid, vectors);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::SearchCentroidIDByVector(float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        faiss::ivflib::search_centroid(faissIndex, vectors, numOfQuery, clusterIDs);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::DeleteVectorsByIDs(size_t numOfQuery, int pids[]) {
    try {
        this->faissIndex->remove_ids(faiss::IDSelectorBatch(numOfQuery, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

int FaissProductClusteringDB::GetVectorTotal() {
    return this->faissIndex->ntotal;
}

void FaissProductClusteringDB::DumpFaissDB(const char fileName[]) {
    try {
        faiss::write_index(this->faissIndex, fileName);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissProductClusteringDB::ResetIndex() {
    try {
        this->faissIndex->reset();
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}
