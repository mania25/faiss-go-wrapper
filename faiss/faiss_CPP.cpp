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
    this->dimension = dimension;
    this->faissIndexType = faissIndexType;
}

void FaissDB::ReadFaissDBFromFile(char *fileName, int ioflags) {
    try {
        this->faissIndex = faiss::read_index(fileName, ioflags);
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissDB::InitFaissDB(int metricType) {
    try {
        this->faissIndex = faiss::index_factory(this->dimension, this->faissIndexType, static_cast<faiss::MetricType>(metricType));
        this->faissIndex->verbose = true;
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissDB::PreAllocateTrainVector(int size) {
    try {
        this->listOfTrainVectors.reserve(this->dimension*size);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissDB::PushTrainDataVector(const float vectors[]) {
    try {
        for (int i = 0; i < (sizeof(*vectors) / sizeof(float)) * this->dimension; ++i) {
            this->listOfTrainVectors.push_back(vectors[i]);
        }
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissDB::ValidateTrainDataset() {
    for (float listOfTrainVector : listOfTrainVectors) {
        float data = listOfTrainVector;

        if (!std::isfinite(data)){
            printf("Invalid vectors data, Got: %f\n", data);
            return;
        }
    }
}

u_long FaissDB::GetTrainDataSize() {
    return static_cast<u_long>(this->listOfTrainVectors.size() / static_cast<u_long>(this->dimension));
}

void FaissDB::BuildIndex(int numOfTrainDataset) {
    try {
        this->faissIndex->train(numOfTrainDataset, this->listOfTrainVectors.data());
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
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
        printf("%s\n", exception.what());
    }
}

void FaissDB::AddNewVectorWithIDs(int sizeOfDatabase, float vectors[], int64_t pids[]) {
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

void FaissDB::SearchVector(int numOfQuery, int nProbe, float *vectors, int kTotal, float *distances, int64_t *pids) {
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

void FaissDB::SearchVectorByID(int64_t pid, int nProbe, float vectors[]) {
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

void FaissDB::SearchCentroidIDByVector(float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
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

void FaissDB::DeleteVectorsByIDs(size_t numOfQuery, int pids[]) {
    try {
        this->faissIndex->remove_ids(faiss::IDSelectorBatch(numOfQuery, reinterpret_cast<const faiss::IDSelector::idx_t *>(pids)));
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

int FaissDB::GetVectorTotal() {
    return this->faissIndex->ntotal;
}

void FaissDB::DumpFaissDB(const char fileName[]) {
    try {
        faiss::write_index(this->faissIndex, fileName);
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}

void FaissDB::ResetIndex() {
    try {
        this->faissIndex->reset();
    } catch (faiss::FaissException &exception) {
        printf("%s\n", exception.what());
    }
}
