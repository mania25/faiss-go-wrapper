//
// Created by Abdurrahman on 21/01/20.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "faiss_C.h"

#define VECTOR_DIMENSION 256

char *removeNewLine( char *s )
{
    char *n = malloc( strlen( s ? s : "\n" ) );
    if( s )
        strcpy( n, s );
    n[strlen(n)-1]='\0';
    return n;
}

void removeString (char text[], int start, int length) {
    int n = strlen(text), i;
    for (i = start+length; i <= n; i++) {
        text[i-length] = text[i];
    }
}

int main() {
    char line[8196];
    char c;
    int count = 0;
    int i = 0;

    printf("Initializing FAISS DB. . .\n");
    FaissProductClusteringDB* faissProductClusteringDb = newFaissProductClusteringDB(256, "OPQ8,IVF10_HNSW32,PQ8");
    InitFaissDB(faissProductClusteringDb);

    printf("Opening CSV file. . .\n");
    FILE *file;
    file = fopen("product_vectors.csv", "r");

    printf("Getting CSV total data. . .\n");
    for (c = getc(file); c != EOF; c = getc(file)) {
        if (c == '\n') { // Increment count if this character is newline
            count = count + 1;
        }
    }
    rewind(file);

    printf("Parsing CSV data. . .\n");
    char *arr[count];
    while (fgets(line, 8196, file))
    {
        char *columns = strtok(line, ";");
        columns = strtok(NULL, ";");

        arr[i] = removeNewLine(columns);
        i++;
    }

    printf("Normalizing CSV data. . .\n");
    for (int k = 0; k < count; ++k) {
        char *data = arr[k];
        if (data == NULL) {
            continue;
        }

        for (int j = 0; j < strlen(data); ++j) {
            if (data[j] == '[' || data[j] == ']' || data[j] == ' ') {
                removeString(data, j, 1);
            }
        }
    }

    printf("Inserting data into FAISS train dataset. . .\n");
    for (int l = 0; l < count; ++l) {
        float vectors[VECTOR_DIMENSION];
        int vectorIndex = 0;

        char *listOfVectors = strtok(arr[l], ",");
        while( listOfVectors != NULL ) {
            vectors[vectorIndex] = strtof(listOfVectors, NULL);
            listOfVectors = strtok(NULL, ",");

            vectorIndex++;
        }

        PushTrainDataVector(faissProductClusteringDb, vectors);
    }
    printf("Query OK; %lu row affected.\n\n", GetTrainDataSize(faissProductClusteringDb));

    printf("Building FAISS index. . .\n");
    BuildIndex(faissProductClusteringDb, count);

    printf("Saving FAISS index. . .");
    DumpFaissDB(faissProductClusteringDb, "product_clusters_10.index");

    return 0;
}