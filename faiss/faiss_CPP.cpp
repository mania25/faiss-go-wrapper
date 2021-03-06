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
    printf("FAISS version %d.%d.%d\n", FAISS_VERSION_MAJOR, FAISS_VERSION_MINOR, FAISS_VERSION_PATCH);

    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}


FaissDB::~FaissDB() {
}

void FaissDB::ReadFaissDBFromFile(char *fileName, int ioflags) {
    try {
        this->faissIndex = faiss::read_index(fileName, ioflags);
    } catch (faiss::FaissException &exception) {
        printf("ReadFaissDBFromFile() : %s\n", exception.what());
        return;
    }

    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->set_direct_map_type(faiss::DirectMap::Hashtable);
    } catch (faiss::FaissException &exception) {
        return;
    }
}

void FaissDB::InitFaissDB(int metricType) {
    try {
        this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, static_cast<faiss::MetricType>(metricType));
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("InitFaissDB() : %s\n", exception.what());
    }

    try {
        faiss::ivflib::extract_index_ivf(faissIndex)->set_direct_map_type(faiss::DirectMap::Hashtable);
    } catch (faiss::FaissException &exception) {
        return;
    }
}

void FaissDB::PreAllocateTrainVector(int size) {
    try {
        unsigned int totalSize = static_cast<unsigned int>(this->dimension)*static_cast<unsigned int>(size);
        this->listOfTrainVectors.reserve(totalSize);
    } catch (std::exception &exception) {
        printf("PreAllocateTrainVector() : %s\n", exception.what());
    }
}

void FaissDB::PushTrainDataVector(const float vectors[]) {
    try {
        for (int i = 0; i < (sizeof(*vectors) / sizeof(float)) * this->dimension; ++i) {
            this->listOfTrainVectors.push_back(vectors[i]);
        }
    } catch (std::exception &exception) {
        printf("PushTrainDataVector() : %s\n", exception.what());
    }
}

void FaissDB::ValidateTrainDataset() {
    for (float listOfTrainVector : listOfTrainVectors) {
        float data = listOfTrainVector;

        if (!std::isfinite(data)){
            printf("ValidateTrainDataset() : Invalid vectors data, Got: %f\n", data);
            return;
        }
    }
}

u_long FaissDB::GetTrainDataSize() {
    auto trainDataSize = static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
    return trainDataSize;
}

void FaissDB::BuildIndex(int numOfTrainDataset) {
    try {
        this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
    } catch (faiss::FaissException &exception) {
        printf("BuildIndex() : %s\n", exception.what());
    }
}

bool FaissDB::GetTrainStatus() {
    bool trainStatus = this->faissIndex->is_trained;
    return trainStatus;
}

void FaissDB::AddNewVector(int sizeOfDatabase, float *vectors) {
    try {
        this->faissIndex->add(sizeOfDatabase, vectors);
    } catch (faiss::FaissException &exception) {
        printf("AddNewVector() : %s\n", exception.what());
    }
}

void FaissDB::AddNewVectorWithIDs(int sizeOfDatabase, float* vectors, int64_t* pids) {
    try {
        this->faissIndex->add_with_ids(sizeOfDatabase, vectors, pids);
    } catch (faiss::FaissException &exception) {
        printf("AddNewVectorWithIDs() : %s\n", exception.what());
    }
}

void FaissDB::SearchVector(int numOfQuery, int nProbe, float *vectors, int kTotal, float *distances, int64_t *pids) {
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
}

void FaissDB::SearchVectorByID(int64_t pid, int nProbe, float vectors[]) {
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
}

void FaissDB::SearchCentroidIDByVector(float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
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
}

void FaissDB::DeleteVectorsByIDs(size_t numOfQuery, int pids[]) {
    try {
        faiss::DirectMap directMap = faiss::ivflib::extract_index_ivf(faissIndex)->direct_map;
        if (directMap.type == faiss::DirectMap::Hashtable) {
            this->faissIndex->remove_ids(faiss::IDSelectorArray(numOfQuery, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
            return;
        }
    } catch (...) {
        printf("DeleteVectorsByIDs() : query not supported for this index, using the default one.\n");
    }

    try {
        this->faissIndex->remove_ids(faiss::IDSelectorBatch(numOfQuery, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
    } catch (faiss::FaissException &exception) {
        printf("DeleteVectorsByIDs() : %s\n", exception.what());
    }
}

int FaissDB::GetVectorTotal() {
    int vectorTotal = this->faissIndex->ntotal;
    return vectorTotal;
}

void FaissDB::DumpFaissDB(const char fileName[]) {
    try {
        if (fileName == nullptr) {
            printf("Invalid fileName. Got: %s\n", fileName);
            return;
        }

        faiss::write_index(this->faissIndex, fileName);
    } catch (faiss::FaissException &exception) {
        printf("DumpFaissDB() : %s\n", exception.what());
    }
}

void FaissDB::ResetIndex() {
    try {
        this->faissIndex->reset();
    } catch (faiss::FaissException &exception) {
        printf("ResetIndex() : %s\n", exception.what());
    }
}
