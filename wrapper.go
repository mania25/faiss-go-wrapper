package main

// #cgo CPPFLAGS: -Ifaiss/include/
// #cgo LDFLAGS: -L${SRCDIR}/faiss/lib -lfaiss-wrapper -lfaiss -lstdc++
// #cgo LDFLAGS: -Wl,-rpath ${SRCDIR}/faiss/lib
//
// typedef void* FaissProductClusteringDB;
// FaissProductClusteringDB getFaissProductClusteringDB( int dimension, int nClusters );
// FaissProductClusteringDB getFaissProductClusteringDB( int dimension, int nClusters );
// void InitFaissDB(FaissProductClusteringDB);
// void BuildIndex(FaissProductClusteringDB);
// void PushTrainDataVector(FaissProductClusteringDB, float vectors[]);
// int GetTrainDataSize(FaissProductClusteringDB);
// void AddNewVector(FaissProductClusteringDB, int sizeOfDatabase, int pids[], float vectorsFloat[]);
// int GetVectorTotal(FaissProductClusteringDB);
import "C"
import (
	"log"
)

type GoFaiss struct {
	faiss C.FaissProductClusteringDB
}

func New() GoFaiss {
	log.Println("Initiating FAISS C++ Object. . .")

	var ret GoFaiss
	ret.faiss = C.getFaissProductClusteringDB(256, 2)
	return ret
}

func (f GoFaiss) InitFaissDB() {
	C.InitFaissDB(f.faiss)
}

func (f GoFaiss) PushTrainDataVector(vectors []C.float)  {
	C.PushTrainDataVector(f.faiss, &vectors[0])
}

func (f GoFaiss) BuildIndex()  {
	log.Println("Building FAISS index. . .")
	C.BuildIndex(f.faiss)
}