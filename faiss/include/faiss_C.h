//
// Created by Abdurrahman on 09/01/20.
//

#ifndef _FAISS_C_H
#define _FAISS_C_H

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
    extern "C"
    {
    #endif

        typedef struct FaissProductClusteringDB FaissProductClusteringDB;
        FaissProductClusteringDB* newFaissProductClusteringDB(int dimension, char *faissIndexType);
        void ReadFaissDBFromFile(FaissProductClusteringDB* fdb, char fileName[]);
        void InitFaissDB(FaissProductClusteringDB* fdb);
        void PushTrainDataVector(FaissProductClusteringDB* fdb, float vectors[]);
        void ValidateTrainDataset(FaissProductClusteringDB* fdb);
        unsigned long GetTrainDataSize(FaissProductClusteringDB* fdb);
        void BuildIndex(FaissProductClusteringDB* fdb, int numOfTrainDataset);
        bool GetTrainStatus(FaissProductClusteringDB* fdb);
        void AddNewVector(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[]);
        void AddNewVectorWithIDs(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[], int64_t pids[]);
        void SearchVector(FaissProductClusteringDB* fdb, int numOfQuery, float vectors[], int kTotal, float distances[], int64_t pids[]);
        void SearchVectorByID(FaissProductClusteringDB* fdb, int64_t pid, float vectors[]);
        void DeleteVectorsByIDs(FaissProductClusteringDB* fdb, int pids[]);
        int GetVectorTotal(FaissProductClusteringDB* fdb);
        void DumpFaissDB(FaissProductClusteringDB* fdb, char fileName[]);
        void ResetIndex(FaissProductClusteringDB* fdb);

    #ifdef __cplusplus
    }
#endif

#endif //_FAISS_C_H
