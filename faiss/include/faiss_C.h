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
        typedef struct FaissProductClusteringDB FaissProductClusteringDB;
        FaissProductClusteringDB* newFaissProductClusteringDB(int dimension, char *faissIndexType);
        void ReadFaissDBFromFile(FaissProductClusteringDB* fdb, char fileName[], int ioflags);
        void InitFaissDB(FaissProductClusteringDB* fdb, int metricType);
        void PushTrainDataVector(FaissProductClusteringDB* fdb, float vectors[]);
        void ValidateTrainDataset(FaissProductClusteringDB* fdb);
        unsigned long GetTrainDataSize(FaissProductClusteringDB* fdb);
        void BuildIndex(FaissProductClusteringDB* fdb, int numOfTrainDataset);
        bool GetTrainStatus(FaissProductClusteringDB* fdb);
        void AddNewVector(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[]);
        void AddNewVectorWithIDs(FaissProductClusteringDB* fdb, int sizeOfDatabase, float vectors[], int64_t pids[]);
        void SearchVector(FaissProductClusteringDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]);
        void SearchVectorByID(FaissProductClusteringDB* fdb, int64_t pid, int nProbe, float vectors[]);
        void SearchCentroidIDByVector(FaissProductClusteringDB* fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs);
        void DeleteVectorsByIDs(FaissProductClusteringDB* fdb, size_t numOfQuery, int pids[]);
        int GetVectorTotal(FaissProductClusteringDB* fdb);
        void DumpFaissDB(FaissProductClusteringDB* fdb, char fileName[]);
        void ResetIndex(FaissProductClusteringDB* fdb);

    #ifdef __cplusplus
    }
#endif

#endif //_FAISS_C_H
