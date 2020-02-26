package main

// 	#cgo CPPFLAGS: -I/usr/local/include
// 	#cgo LDFLAGS: -L/usr/local/lib -lfaiss-wrapper -lfaiss -lstdc++
// 	#cgo LDFLAGS: -Wl,-rpath /usr/local/lib
//
//  #include <stdbool.h>
//  #include <stdlib.h>
//
//  #if __linux__
//      #include <cstdint>
//  #elif __APPLE__
//      #include <stdint.h>
//  #else
//      #error "Platform not supported"
//  #endif
//
//  typedef struct FaissDB FaissDB;
//  FaissDB* newFaissDB(int dimension, char *faissIndexType);
//  void ReadFaissDBFromFile(FaissDB* fdb, char fileName[], int ioflags);
//  void InitFaissDB(FaissDB* fdb, int metricType);
//  void PushTrainDataVector(FaissDB* fdb, float vectors[]);
//  void ValidateTrainDataset(FaissDB* fdb);
//  unsigned long GetTrainDataSize(FaissDB* fdb);
//  void BuildIndex(FaissDB* fdb, int numOfTrainDataset);
//  bool GetTrainStatus(FaissDB* fdb);
//  void AddNewVector(FaissDB* fdb, int sizeOfDatabase, float vectors[]);
//  void AddNewVectorWithIDs(FaissDB* fdb, int sizeOfDatabase, float vectors[], int64_t pids[]);
//  void SearchVector(FaissDB* fdb, int numOfQuery, int nProbe, float vectors[], int kTotal, float distances[], int64_t pids[]);
//  void SearchVectorByID(FaissDB* fdb, int64_t pid, int nProbe, float vectors[]);
//  void SearchCentroidIDByVector(FaissDB* fdb, float *vectors, int numOfQuery, int nProbe, int64_t *clusterIDs);
//  void DeleteVectorsByIDs(FaissDB* fdb, size_t numOfQuery, int pids[]);
//  int GetVectorTotal(FaissDB* fdb);
//  void DumpFaissDB(FaissDB* fdb, char fileName[]);
//  void ResetIndex(FaissDB* fdb);
import "C"
import (
	"log"
)

type GoFaiss struct {
	faiss *C.FaissDB
}

func New() GoFaiss {
	log.Println("Initiating FAISS C++ Object. . .")

	var ret GoFaiss
	ret.faiss = C.newFaissDB(256, C.CString("Flat"))
	return ret
}

func (f GoFaiss) InitFaissDB() {
	C.InitFaissDB(f.faiss, C.int(1))
}

func (f GoFaiss) PushTrainDataVector(vectors []C.float) {
	C.PushTrainDataVector(f.faiss, &vectors[0])
}

func (f GoFaiss) BuildIndex(totalTrainDataset int) {
	log.Println("Building FAISS index. . .")

	cTotalTrainDataset := (C.int)(totalTrainDataset)
	C.BuildIndex(f.faiss, cTotalTrainDataset)
}
