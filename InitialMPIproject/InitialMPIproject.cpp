// InitialMPIproject.cpp : Defines the entry point for the console application.
//

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/
#define MAX_POINTS 500000
#define MAX_DIMENTIONS 20

const char* INPUT_FILE = "C:\\ParallelProject\\input.txt";
const char* OUTPUT_FILE = "C:\\ParallelProject\\output.txt";

/* This is an interactive version of cpi */
#include <mpi.h>
#include <stdio.h>
//#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <math.h>

//using namespace std;

//•	N - number of points
//•	K – number of coordinates of points
//•	Coordinates of all points with attached value : 1 for those that belong to set A and -1 for the points that belong to set B.
//•	alphaZero – increment value of alpha, alphaMax – maximum value of alpha
//•	LIMIT – the maximum number of iterations.
//•	QC – Quality of Classifier to be reached

typedef struct {
	float * x;
	int set;
} POINT;


int readPointFromFile(FILE * fp, POINT * points, int pointIndex, int K)
{
	if (feof(fp))
	{
		return 0;
	}

	for (int i = 0; i < K; i++)
	{
		if (!fscanf(fp, "%f ", &(points[pointIndex].x[i])))
			return 0;
	}
	points[pointIndex].x[K] = 1;
	if (!fscanf(fp, "%d \n", &(points[pointIndex].set)))
		return 0;
	return 1;
}

int sign(float num)
{
	if (num >= 0)
		return 1;
	else
		return -1;
}

float func(float * x, float * weights, int K)
{
	float result = 0;
	for (int i = 0; i < K + 1; i++)
	{
		result += weights[i] * x[i];
	}
	return result;
}

void updateWeights(float * x, float * weights, int signResult, float alpha, int K)
{
	for (int i = 0; i < K + 1; i++)
	{
		weights[i] = weights[i] + alpha*-signResult*x[i];
	}
}

void zeroWeights(float ** weights, int K, int numOfWeights)
{
	for (int i = 0; i < numOfWeights; i++)
	{
		memset(weights[i], 0, sizeof(float)*(K + 1));
	}
}

int isMiss(int &signResult, POINT * points, int i, float * weights, int K, int &Nmis)
{
	signResult = sign(func(points[i].x, weights, K));
	if (signResult != points[i].set)
	{
		return 1;
	}
	return 0;
}

void freeMemory(int N, POINT * points, int numOfSegments, float ** weights, bool * isSegmentClassifiedProperly, char * buffer)
{
	for (int i = 0; i < N; i++)
	{
		free(points[i].x);
	}
	for (int i = 0; i < numOfSegments; i++)
	{
		free(weights[i]);
	}
	free(weights);
	free(points);
	free(isSegmentClassifiedProperly);
	free(buffer);
}

int main(int argc, char *argv[])
{
	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Get_processor_name(processor_name, &namelen);

	MPI_Status status;

	float alpha, alphaZero, alphaMax, QC, quality;
	int N, Nmis = 0, K, LIMIT, pointNum = 0;
	bool isAlphaFound = false;
	double t1, t2;
	POINT * points;
	FILE * fp;
	float ** weights;
	int numOfSegments = 16;
	char * buffer;

	//if (numprocs < 2)
	//{
	//	printf("Number of processes must be greater than 1\n");
	//	MPI_Abort(MPI_COMM_WORLD, 0);
	//}

	if (myid == 0)
	{
		fp = fopen(INPUT_FILE, "r");
		if (fp == NULL)
		{
			printf("file could not be opened for reading\n");
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		// N    K    alphaZero   alphaMax LIMIT   QC
		fscanf(fp, "%d %d %f %f %d %f \n", &N, &K, &alphaZero, &alphaMax, &LIMIT, &QC);
		//MPI_Recv(&answer, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaZero, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&LIMIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&QC, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	points = (POINT*)malloc(sizeof(POINT) * N);
	weights = (float**)malloc(sizeof(float*) * numOfSegments);
	bool * isSegmentClassifiedProperly = (bool*)calloc(numOfSegments, sizeof(bool));
	for (int i = 0; i < numOfSegments; i++)
	{
		weights[i] = (float*)calloc(K + 1, sizeof(float));
	}

	for (int i = 0; i < N; i++)
	{
		points[i].x = (float*)malloc(sizeof(float) * (K + 1));
	}

	buffer = (char*)malloc(sizeof(float) * N * (K + 2));
	int BUFFER_SIZE = sizeof(float) * N * (K + 2);
	int position;

	if (myid == 0)
	{
		for (int i = 0; i < N; i++)
		{
			readPointFromFile(fp, points, i, K);
		}
		fclose(fp);
	}

	t1 = MPI_Wtime();

	if (myid == 0)
	{
		position = 0;
		for (int i = 0; i < N; i++)
		{
			MPI_Pack(points[i].x, K + 1, MPI_FLOAT, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);
			MPI_Pack(&points[i].set, 1, MPI_INT, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);
		}
	}
	MPI_Bcast(&position, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(buffer, position, MPI_PACKED, 0, MPI_COMM_WORLD);
	if (myid != 0)
	{
		position = 0;
		for (int i = 0; i < N; i++)
		{
			MPI_Unpack(buffer, BUFFER_SIZE, &position, points[i].x, K + 1, MPI_FLOAT, MPI_COMM_WORLD);
			MPI_Unpack(buffer, BUFFER_SIZE, &position, &points[i].set, 1, MPI_INT, MPI_COMM_WORLD);
		}
	}

	//for (int i = 0; i < N; i++) 
	//{
	//	MPI_Bcast(points[i].x, K + 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//	MPI_Bcast(&points[i].set, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//}

	

	// Give each process an alpha range to work on
	float alpha_low = alphaZero + myid * ((alphaMax-alphaZero) / numprocs);
	float alpha_high = alphaZero + (myid + 1) * ((alphaMax - alphaZero) / numprocs);
	//QC = 0.01f;
	for (alpha = alpha_low; alpha < alpha_high; alpha += alphaZero)
	{
		zeroWeights(weights, K, numOfSegments);
		bool isClassifiedProperly = false;
		int iterCounter = 0;

		#pragma omp parallel
		{
			while (!isClassifiedProperly && iterCounter < LIMIT)
			{
				#pragma omp for
				for (int i = 0; i < numOfSegments; i++)
				{
					isSegmentClassifiedProperly[i] = true;
					int start = i * (N / numOfSegments);
					int finish = i == numOfSegments - 1 ? N : (i + 1) * (N / numOfSegments);
					for (int j = start; j < finish; j++)
					{
						if (isSegmentClassifiedProperly[i])
						{
							int signResult = sign(func(points[i].x, weights[i], K));
							if (signResult != points[i].set)
							{
								updateWeights(points[i].x, weights[i], signResult, alpha, K);
								isSegmentClassifiedProperly[i] = false;
								break;
							}
						}
					}
				}

				#pragma omp single
				{
					for (int i = 1; i < numOfSegments; i++) // Averaging the weights
					{
						for (int j = 0; j < K + 1; j++)
						{
							weights[0][j] += (weights[i][j] - weights[0][j]) / (i + 1);
						}
					}

					for (int i = 1; i < numOfSegments; i++) // Reset all weights to be the same as the averaged weight 0
					{
						memcpy(weights[i], weights[0], sizeof(float)*(K + 1));
					}
					isClassifiedProperly = true;
					for (int i = 0; i < numOfSegments; i++) // Check if every segment is classified properly, if so then end
					{
						if (isSegmentClassifiedProperly[i] == false)
						{
							isClassifiedProperly = false;
							break;
						}
					}
					iterCounter++;
				}
			}
		}

		Nmis = 0;
		#pragma omp parallel for reduction(+:Nmis)
		for (int i = 0; i < N; i++) // Count number of misses for evaluation of quality 
		{
			int signResult = 0;
			Nmis += isMiss(signResult, points, i, weights[0], K, Nmis);
		}

		quality = (float)Nmis / N;

		if (quality < QC)
		{
			isAlphaFound = true;
			break;
		}
		//alpha += alphaZero;
	}
	float min_quality = 0;

	MPI_Reduce(&quality, &min_quality, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

	//quality = min_quality;

	t2 = MPI_Wtime();

	if (myid == 0)
	{
		fp = fopen(OUTPUT_FILE, "w");
		if (fp == NULL)
		{
			printf("file could not be opened for writing\n");
			freeMemory(N, points, numOfSegments, weights, isSegmentClassifiedProperly, buffer);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		//// Print results to CMD 
		//if (isAlphaFound)
		//{
		//	printf("Alpha minimum: %1.4f\n", alpha);
		//	for (int i = 0; i < K + 1; i++)
		//	{
		//		printf("W%d: %1.4f\n", i + 1, weights[0][i]);
		//	}
		//	printf("q: %1.5f\n", quality);

		//	printf("\nMPI_Wtime: %1.6f seconds\n", t2 - t1);
		//}
		//else
		//{
		//	printf("Alpha is not found\n");
		//}

		// Print results to output.txt
		if (isAlphaFound)
		{
			fprintf(fp, "Alpha minimum: %1.4f\n", alpha);
			for (int i = 0; i < K + 1; i++)
			{
				fprintf(fp, "%1.4f\n", weights[0][i]);
			}
			fprintf(fp, "%1.5f\n", quality);

			fprintf(fp, "\nMPI_Wtime: %1.6f seconds\n", t2 - t1);// fflush(stdout);
		}
		else
		{
			fprintf(fp, "Alpha is not found\n");
		}

		fclose(fp);
	}

	// Print results to CMD 
	if (isAlphaFound)

	{
		printf("\nAlpha minimum: %1.4f\n", alpha);
		for (int i = 0; i < K + 1; i++)
		{
			printf("W%d: %1.4f\n", i + 1, weights[0][i]);
		}
		printf("q: %1.5f\n", quality);

		printf("\nMPI_Wtime: %1.6f seconds\n", t2 - t1);
	}
	else
	{
		printf("Alpha is not found\n");
	}

	// free memory
	freeMemory(N, points, numOfSegments, weights, isSegmentClassifiedProperly, buffer);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
	return 0;
}

//#include <stdio.h>
//#include <omp.h>
//#include <vector>
//
//using namespace std;
//
//int segment_read(char *buff, const int len, const int count) {
//	return 1;
//}
//
//void foo(char* buffer, size_t size) {
//	int count_of_reads = 0;
//	int count = 1;
//	std::vector<int> *posa;
//	int nthreads;
//
//#pragma omp parallel 
//	{
//		nthreads = omp_get_num_threads();
//		const int ithread = omp_get_thread_num();
//#pragma omp single 
//		{
//			posa = new vector<int>[nthreads];
//			posa[0].push_back(0);
//		}
//
//		//get the number of lines and end of line position
//#pragma omp for reduction(+: count)
//		for (int i = 0; i < size; i++) {
//			if (buffer[i] == '\n') { //should add EOF as well to be safe
//				count++;
//				posa[ithread].push_back(i);
//			}
//		}
//
//#pragma omp for     
//		for (int i = 1; i < count; i++) {
//			const int len = posa[ithread][i] - posa[ithread][i - 1];
//			char* buff = &buffer[posa[ithread][i - 1]];
//			const int sequence_counter = segment_read(buff, len, i);
//			if (sequence_counter == 1) {
//#pragma omp atomic
//				count_of_reads++;
//				printf("\n Total No. of reads: %d \n", count_of_reads);
//			}
//
//		}
//	}
//	delete[] posa;
//}
//
//int main() {
//	FILE * pFile;
//	long lSize;
//	char * buffer;
//	size_t result;
//
//	pFile = fopen("myfile.txt", "rb");
//	if (pFile == NULL) { fputs("File error", stderr); exit(1); }
//
//	// obtain file size:
//	fseek(pFile, 0, SEEK_END);
//	lSize = ftell(pFile);
//	rewind(pFile);
//
//	// allocate memory to contain the whole file:
//	buffer = (char*)malloc(sizeof(char)*lSize);
//	if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }
//
//	// copy the file into the buffer:
//	result = fread(buffer, 1, lSize, pFile);
//	if (result != lSize) { fputs("Reading error", stderr); exit(3); }
//
//	/* the whole file is now loaded in the memory buffer. */
//	foo(buffer, result);
//	// terminate
//
//
//	fclose(pFile);
//	free(buffer);
//	return 0;
//}