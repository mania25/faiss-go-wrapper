//
// Created by Abdurrahman on 07/01/20.
//

#include <math.h>
#include "faiss_CPP.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/impl/AuxIndexStructures.h"
#include <faiss/IVFlib.h>

FaissDB::FaissDB(int dimension, const char *faissIndexType) {
    mtx.lock();
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
    mtx.unlock();
}


FaissDB::~FaissDB() {
    printf("Destructor Called.\n");
}

void FaissDB::ReadFaissDBFromFile(char *fileName, int ioflags) {
    mtx.lock();
    try {
        this->faissIndex = faiss::read_index(fileName, ioflags);
    } catch (faiss::FaissException &exception) {
        printf("ReadFaissDBFromFile() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::InitFaissDB(int metricType) {
    mtx.lock();
    try {
        this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, static_cast<faiss::MetricType>(metricType));
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("InitFaissDB() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::PreAllocateTrainVector(int size) {
    mtx.lock();
    try {
        unsigned int totalSize = static_cast<unsigned int>(this->dimension)*static_cast<unsigned int>(size);
        this->listOfTrainVectors.reserve(totalSize);
    } catch (std::exception &exception) {
        printf("PreAllocateTrainVector() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::PushTrainDataVector(const float vectors[]) {
    mtx.lock();
    try {
        for (int i = 0; i < (sizeof(*vectors) / sizeof(float)) * this->dimension; ++i) {
            this->listOfTrainVectors.push_back(vectors[i]);
        }
    } catch (std::exception &exception) {
        printf("PushTrainDataVector() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::ValidateTrainDataset() {
    mtx.lock();
    for (float listOfTrainVector : listOfTrainVectors) {
        float data = listOfTrainVector;

        if (!std::isfinite(data)){
            printf("ValidateTrainDataset() : Invalid vectors data, Got: %f\n", data);
            return;
        }
    }
    mtx.unlock();
}

u_long FaissDB::GetTrainDataSize() {
    mtx.lock();
    auto trainDataSize = static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
    mtx.unlock();

    return trainDataSize;
}

void FaissDB::BuildIndex(int numOfTrainDataset) {
    mtx.lock();
    try {
        this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
    } catch (faiss::FaissException &exception) {
        printf("BuildIndex() : %s\n", exception.what());
    }
    mtx.unlock();
}

bool FaissDB::GetTrainStatus() {
    mtx.lock();
    bool trainStatus = this->faissIndex->is_trained;
    mtx.unlock();

    return trainStatus;
}

void FaissDB::AddNewVector(int sizeOfDatabase, float *vectors) {
    mtx.lock();
    try {
        this->faissIndex->add(sizeOfDatabase, vectors);
    } catch (faiss::FaissException &exception) {
        printf("AddNewVector() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::AddNewVectorWithIDs(int sizeOfDatabase, float* vectors, int64_t* pids) {
    mtx.lock();
    try {
        if (vectors == nullptr || pids == nullptr || !this || this->faissIndex == nullptr) {
            if (vectors == nullptr) {
                printf("`vectors` param is null\n");
            }

            if (pids == nullptr) {
                printf("`pids` param is null\n");
            }

            if (!this) {
                printf("`this` keyword is null\n");
            }

            if (this->faissIndex == nullptr) {
                printf("`this->faissIndex` param is null\n");
            }

            return;
        }

        this->faissIndex->add_with_ids(sizeOfDatabase, vectors, pids);
    } catch (faiss::FaissException &exception) {
        printf("AddNewVectorWithIDs() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::SearchVector(int numOfQuery, int nProbe, float *vectors, int kTotal, float *distances, int64_t *pids) {
    mtx.lock();
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("SearchVector() : nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        this->faissIndex->search(numOfQuery, vectors, kTotal, distances, pids);
    } catch (faiss::FaissException &exception) {
        printf("SearchVector() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::SearchVectorByID(int64_t pid, int nProbe, float vectors[]) {
    mtx.lock();
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("SearchVectorByID() : nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        this->faissIndex->reconstruct(pid, vectors);
    } catch (faiss::FaissException &exception) {
        printf("SearchVectorByID() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::SearchCentroidIDByVector(float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
    mtx.lock();
    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->nprobe = nProbe;
    } catch (...) {
        printf("SearchCentroidIDByVector() : nprobe is not supported in %s index\n", faissIndexType);
    }

    try {
        faiss::ivflib::search_centroid(faissIndex, vectors, numOfQuery, clusterIDs);
    } catch (faiss::FaissException &exception) {
        printf("SearchCentroidIDByVector() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::DeleteVectorsByIDs(size_t numOfQuery, int pids[]) {
    mtx.lock();
    try {
        this->faissIndex->remove_ids(faiss::IDSelectorBatch(numOfQuery, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
    } catch (faiss::FaissException &exception) {
        printf("DeleteVectorsByIDs() : %s\n", exception.what());
    }
    mtx.unlock();
}

int FaissDB::GetVectorTotal() {
    mtx.lock();
    int vectorTotal = this->faissIndex->ntotal;
    mtx.unlock();

    return vectorTotal;
}

void FaissDB::DumpFaissDB(const char fileName[]) {
    mtx.lock();
    try {
        if (fileName == nullptr) {
            printf("Invalid fileName. Got: %s\n", fileName);
            return;
        }

        faiss::write_index(this->faissIndex, fileName);
    } catch (faiss::FaissException &exception) {
        printf("DumpFaissDB() : %s\n", exception.what());
    }
    mtx.unlock();
}

void FaissDB::ResetIndex() {
    mtx.lock();
    try {
        this->faissIndex->reset();
    } catch (faiss::FaissException &exception) {
        printf("ResetIndex() : %s\n", exception.what());
    }
    mtx.unlock();
}
