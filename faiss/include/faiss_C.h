//
// Created by Abdurrahman on 09/01/20.
//

#ifndef _FAISS_C_H
#define _FAISS_C_H

#ifdef __cplusplus
    extern "C"
    {
    #endif

        struct vectorResult {
            float *distance;
            float *pids;
        };
        typedef void* FaissProductClusteringDB;
        FaissProductClusteringDB getFaissProductClusteringDB(int dimension, char *faissIndexType);
        void ReadFaissDBFromFile(FaissProductClusteringDB, char fileName[]);
        void InitFaissDB(FaissProductClusteringDB);
        void BuildIndex(FaissProductClusteringDB);
        void PushTrainDataVector(FaissProductClusteringDB, float vectors[]);
        int GetTrainDataSize(FaissProductClusteringDB);
        void AddNewVector(FaissProductClusteringDB, int sizeOfDatabase, int pids[], float vectorsFloat[]);
        vectorResult SearchVector(FaissProductClusteringDB, int numOfQuery, float vectors[], int kTotal);
        void DeleteVectorsByIDs(FaissProductClusteringDB, int pids[]);
        int GetVectorTotal(FaissProductClusteringDB);
        void DumpFaissDB(FaissProductClusteringDB, char fileName[]);
        void ResetIndex(FaissProductClusteringDB);

    #ifdef __cplusplus
    }
#endif

#endif //_FAISS_C_H
