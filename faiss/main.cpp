//
// Created by Abdurrahman on 21/01/20.
//

#include <faiss_CPP.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <fstream>
#include <boost/algorithm/string.hpp>

int main() {
    int count = 0;
    const char *indexType = "OPQ8,IVF10_HNSW32,PQ8";
    FaissProductClusteringDB productClusteringDb(256, indexType);

    std::ifstream file("product_vectors.csv");
    std::string line;

    std::cout << "Initializing FAISS DB. . ." << std::endl;
    productClusteringDb.InitFaissDB();

    std::cout << std::endl;

    std::cout << "Adding product vectors as train dataset. . ." << std::endl;
    while (getline(file, line))
    {
        count++;

        std::vector<std::string> vec;
        std::vector<std::string> pVec;
        std::vector<float> pVecFloat;

        boost::algorithm::split(vec, line, boost::is_any_of(";"));

        auto rowData = vec.data();
        auto pidString = rowData[0].data();

        rowData[1].erase(std::remove(rowData[1].begin(), rowData[1].end(), '['), rowData[1].end());
        rowData[1].erase(std::remove(rowData[1].begin(), rowData[1].end(), ']'), rowData[1].end());
        auto vectorString = rowData[1].data();

        boost::algorithm::split(pVec, vectorString, boost::is_any_of(","));
        pVecFloat.reserve(pVec.size());

        for (auto & i : pVec) {
            i.erase(std::remove(i.begin(), i.end(), ' '), i.end());
            pVecFloat.push_back(std::stof(i));
        }

        productClusteringDb.PushTrainDataVector(pVecFloat.data());
    }
    printf("Query OK; %lu rows affected.\n", productClusteringDb.GetTrainDataSize());

    std::cout << std::endl;

    std::cout << "Building FAISS index. . ." << std::endl;
    productClusteringDb.BuildIndex(count);
    std::cout << "Building FAISS index finished." << std::endl;

    std::cout << std::endl;

    std::cout << "Saving FAISS index. . ." << std::endl;
    const char* indexFileName = "product_clusters_10.index";
    productClusteringDb.DumpFaissDB(indexFileName);
    std::cout << "Save FAISS index finished." << std::endl;

    file.close();
    return 0;
}