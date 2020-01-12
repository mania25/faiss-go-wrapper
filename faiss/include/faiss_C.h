//
// Created by Abdurrahman on 09/01/20.
//

#ifndef _FAISS_C_H
#define _FAISS_C_H

#include <cstdint>

#ifdef __cplusplus
    extern "C"
    {
    #endif

        typedef void* FaissProductClusteringDB;
        FaissProductClusteringDB getFaissProductClusteringDB(int dimension, char *faissIndexType);
        void ReadFaissDBFromFile(FaissProductClusteringDB, char fileName[]);
        void InitFaissDB(FaissProductClusteringDB);
        void BuildIndex(FaissProductClusteringDB);
        void PushTrainDataVector(FaissProductClusteringDB, float vectors[]);
        int GetTrainDataSize(FaissProductClusteringDB);
        void AddNewVector(FaissProductClusteringDB, int sizeOfDatabase, int pids[], float vectorsFloat[]);
        void SearchVector(FaissProductClusteringDB, int numOfQuery, float vectors[], int kTotal, float distances[], int64_t pids[]);
        void DeleteVectorsByIDs(FaissProductClusteringDB, int pids[]);
        int GetVectorTotal(FaissProductClusteringDB);
        void DumpFaissDB(FaissProductClusteringDB, char fileName[]);
        void ResetIndex(FaissProductClusteringDB);

    #ifdef __cplusplus
    }
#endif

#endif //_FAISS_C_H
