//
// Created by Abdurrahman on 09/01/20.
//

#ifndef _FAISS_C_H
#define _FAISS_C_H

    #ifdef __cplusplus
    extern "C"
    {
    #endif

        typedef void* FaissProductClusteringDB;
        FaissProductClusteringDB getFaissProductClusteringDB( int dimension, int nClusters );
        void InitFaissDB(FaissProductClusteringDB);
        void BuildIndex(FaissProductClusteringDB);
        void PushTrainDataVector(FaissProductClusteringDB, float vectors[]);
        int GetTrainDataSize(FaissProductClusteringDB);
        void AddNewVector(FaissProductClusteringDB, int sizeOfDatabase, int pids[], float vectorsFloat[]);
        int GetVectorTotal(FaissProductClusteringDB);
        void DumpFaissDB(FaissProductClusteringDB, char fileName[]);

    #ifdef __cplusplus
    }
    #endif

#endif //_FAISS_C_H
