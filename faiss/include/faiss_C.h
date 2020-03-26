//
// Created by Abdurrahman on 09/01/20.
//

#ifndef _FAISS_C_H
#define _FAISS_C_H

#include <stdbool.h>
#include <stdlib.h>

#if __linux__
#include <cstdint>
#elif __APPLE__
#include <stdint.h>
#else
#error "Platform not supported"
#endif

#ifdef __cplusplus
    extern "C"
    {
    #endif

        enum MetricType {
            METRIC_INNER_PRODUCT = 0,  ///< maximum inner product search
            METRIC_L2 = 1,             ///< squared L2 search
            METRIC_L1,                 ///< L1 (aka cityblock)
            METRIC_Linf,               ///< infinity distance
            METRIC_Lp,                 ///< L_p distance, p is given by metric_arg

            /// some additional metrics defined in scipy.spatial.distance
            METRIC_Canberra = 20,
            METRIC_BrayCurtis,
            METRIC_JensenShannon,

        };
        typedef struct FaissDB FaissDB;
        FaissDB* newFaissDB(int dimension, char *faissIndexType);
        void ReadFaissDBFromFile(FaissDB* fdb, char fileName[], int ioflags);
        void InitFaissDB(FaissDB* fdb, int metricType);
        void PreAllocateTrainVector(FaissDB* fdb, int size);
        void PushTrainDataVector(FaissDB* fdb, float vectors[]);
        void ValidateTrainDataset(FaissDB* fdb);
        unsigned long GetTrainDataSize(FaissDB* fdb);
        void BuildIndex(FaissDB* fdb, int numOfTrainDataset);
        bool GetTrainStatus(FaissDB* fdb);
        void AddNewVector(FaissDB* fdb, int sizeOfDatabase, float vectors[]);
        void AddNewVectorWithIDs(FaissDB* fdb, int sizeOfDatabase, float* vectors, long long int* pids);
        void SearchVector(FaissDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]);
        void SearchVectorByID(FaissDB* fdb, int64_t pid, int nProbe, float vectors[]);
        void SearchCentroidIDByVector(FaissDB* fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs);
        void DeleteVectorsByIDs(FaissDB* fdb, size_t numOfQuery, int pids[]);
        int GetVectorTotal(FaissDB* fdb);
        void DumpFaissDB(FaissDB* fdb, char fileName[]);
        void ResetIndex(FaissDB* fdb);

    #ifdef __cplusplus
    }
#endif

#endif //_FAISS_C_H
