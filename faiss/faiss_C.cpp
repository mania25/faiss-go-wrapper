//
// Created by Abdurrahman on 09/01/20.
//

#include "faiss_C.h"
#include "faiss_CPP.h"

FaissDB* newFaissDB(int dimension, char *faissIndexType) {
    return new FaissDB(dimension, faissIndexType);
}

void ReadFaissDBFromFile(FaissDB* fdb, char fileName[], int ioflags) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->ReadFaissDBFromFile(fileName, ioflags);
}

void InitFaissDB(FaissDB* fdb, int metricType) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->InitFaissDB(metricType);
}

void PreAllocateTrainVector(FaissDB* fdb, int size) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->PreAllocateTrainVector(size);
}

void PushTrainDataVector(FaissDB* fdb, float vectors[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->PushTrainDataVector(vectors);
}

void ValidateTrainDataset(FaissDB* fdb) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->ValidateTrainDataset();
}

unsigned long GetTrainDataSize(FaissDB* fdb) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return 0;
    }

    return fdb->GetTrainDataSize();
}

void BuildIndex(FaissDB* fdb, int numOfTrainDataset) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->BuildIndex(numOfTrainDataset);
}

bool GetTrainStatus(FaissDB* fdb) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return false;
    }

    return fdb->GetTrainStatus();
}

void AddNewVector(FaissDB* fdb, int sizeOfDatabase, float vectors[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->AddNewVector(sizeOfDatabase, vectors);
}

void AddNewVectorWithIDs(FaissDB* fdb, int sizeOfDatabase, float* vectors, int64_t* pids) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->AddNewVectorWithIDs(sizeOfDatabase, vectors, pids);
}

void SearchVector(FaissDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->SearchVector(numOfQuery, nProbe, vectors, kTotal, distances, pids);
}

void SearchVectorByID(FaissDB* fdb, int64_t pid, int nProbe, float vectors[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->SearchVectorByID(pid, nProbe, vectors);
}

void SearchCentroidIDByVector(FaissDB *fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->SearchCentroidIDByVector(vectors, numOfQuery, nProbe, clusterIDs);
}

void DeleteVectorsByIDs(FaissDB* fdb, size_t numOfQuery, int pids[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->DeleteVectorsByIDs(numOfQuery, pids);
}

int GetVectorTotal(FaissDB* fdb) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return 0;
    }

    return fdb->GetVectorTotal();
}

void DumpFaissDB(FaissDB* fdb, char fileName[]) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->DumpFaissDB(fileName);
}

void ResetIndex(FaissDB* fdb) {
    if (fdb == nullptr) {
        printf("FAISS DB Client is null.\n");
        return;
    }

    fdb->ResetIndex();
}
