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

float x[MAX_POINTS][MAX_DIMENTIONS+1] = { 0 };
int y[MAX_POINTS] = { 0 };

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

int readPointFromFileXY(FILE * fp, float * x, int * y, int pointIndex, int K)
{
	if (feof(fp))
	{
		return 0;
	}

	for (int i = 0; i < K; i++)
	{
		if (!fscanf(fp, "%f ", &x[i]))
			return 0;
	}
	x[K] = 1;
	if (!fscanf(fp, "%d \n", &y[pointIndex]))
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

int isMiss(POINT * points, int index, float * weights, int K, int &Nmis)
{
	int signResult = sign(func(points[index].x, weights, K));
	if (signResult != points[index].set)
	{
		return 1;
	}
	return 0;
}

int isMissXY(float * x, int * y, int index, float * weights, int K, int &Nmis)
{
	int signResult = sign(func(&x[index], weights, K));
	if (signResult != y[index])
	{
		return 1;
	}
	return 0;
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
	int isAlphaFound = false;
	double t1, t2;
	POINT * points;
	FILE * fp;
	float ** weights;
	int numOfSegments = 16;
	int * isAlphaFoundArr;
	int processWithMinAlpha = -1;

	if (numprocs < 2)
	{
		printf("Number of processes must be greater than 1\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (myid == 0)
	{
		fp = fopen(INPUT_FILE, "r");
		if (fp == NULL)
		{
			printf("file could not be opened for reading\n");
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		fscanf(fp, "%d %d %f %f %d %f \n", &N, &K, &alphaZero, &alphaMax, &LIMIT, &QC);
	}


	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaZero, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&LIMIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&QC, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	points = (POINT*)malloc(sizeof(POINT) * N);
	weights = (float**)malloc(sizeof(float*) * numOfSegments);
	isAlphaFoundArr = (int*)calloc(numprocs, sizeof(int));
	bool * isSegmentClassifiedProperly = (bool*)calloc(numOfSegments, sizeof(bool));
	for (int i = 0; i < numOfSegments; i++)
	{
		weights[i] = (float*)calloc(K + 1, sizeof(float));
	}

	for (int i = 0; i < N; i++)
	{
		points[i].x = (float*)malloc(sizeof(float) * (K + 1));
	}

	t1 = MPI_Wtime();

	for (int i = 0; i < N; i++)
	{
		if (myid == 0)
		{
			readPointFromFile(fp, points, i, K);
		}
		MPI_Bcast(points[i].x, K + 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&points[i].set, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	if (myid == 0)
	{
		fclose(fp);
	}

	// Give each process an alpha range to work on
	float alpha_low = alphaZero + myid * ((alphaMax-alphaZero) / numprocs);
	float alpha_high = alphaZero + (myid + 1) * ((alphaMax - alphaZero) / numprocs);

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
							//int signResult = sign(func(x[j], weights[i], K));
							//if (signResult != y[j])
							int signResult = sign(func(points[j].x, weights[i], K));
							if (signResult != points[j].set)
							{
								//updateWeights(x[j], weights[i], signResult, alpha, K);
								updateWeights(points[j].x, weights[i], signResult, alpha, K);
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
			Nmis += isMiss(points, i, weights[0], K, Nmis);
			//Nmis += isMissXY(x[i], y, i, weights[0], K, Nmis);
		}

		quality = (float)Nmis / N;

		if (quality < QC)
		{
			isAlphaFound = true;
			break;
		}
	}

	// Gather results from each process
	float * final_weights = (float*)calloc(numprocs * (K + 1), sizeof(float));
	float * alpha_buf = (float*)calloc(numprocs, sizeof(float));
	float * quality_buf = (float*)calloc(numprocs, sizeof(float));

	MPI_Gather(&isAlphaFound, 1, MPI_INT, isAlphaFoundArr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&alpha, 1, MPI_FLOAT, alpha_buf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(&quality, 1, MPI_FLOAT, quality_buf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(weights[0], K + 1, MPI_FLOAT, final_weights, K + 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	t2 = MPI_Wtime();

	if (myid == 0)
	{
		// Find lowest value of viable alpha from other processes
		processWithMinAlpha = -1;
		for (int i = 0; i < numprocs; i++)
		{
			if (isAlphaFoundArr[i])
			{
				processWithMinAlpha = i;
				break;
			}
		}

		fp = fopen(OUTPUT_FILE, "w");
		if (fp == NULL)
		{
			printf("file could not be opened for writing\n");
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		// Print results to CMD 
		if (processWithMinAlpha != -1)
		{
			printf("Alpha minimum: %1.4f\n", alpha_buf[processWithMinAlpha]);
			for (int i = 0; i < K + 1; i++)
			{
				printf("%1.4f\n", final_weights[i + processWithMinAlpha*(K+1)]);
			}
			printf("q: %1.5f\n", quality_buf[processWithMinAlpha]);

			printf("\nMPI_Wtime: %1.6f seconds\n", t2 - t1);
		}
		else
		{
			printf("Alpha is not found\n");
		}

		// Print results to output.txt
		if (processWithMinAlpha != -1)
		{
			fprintf(fp, "Alpha minimum: %1.4f\n", alpha_buf[processWithMinAlpha]);
			for (int i = 0; i < K + 1; i++)
			{
				fprintf(fp, "%1.4f\n", final_weights[i + processWithMinAlpha*(K + 1)]);
			}
			fprintf(fp, "%1.5f\n", quality_buf[processWithMinAlpha]);

			fprintf(fp, "\nMPI_Wtime: %1.6f seconds\n", t2 - t1);// fflush(stdout);
		}
		else
		{
			fprintf(fp, "Alpha is not found\n");
		}

		fclose(fp);
	}

	printf("\nProcess #%d MPI_Wtime: %1.6f seconds\n", myid, t2 - t1);

	// free memory
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
	free(isAlphaFoundArr);
	free(final_weights);
	free(alpha_buf);
	free(quality_buf);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
	return 0;
}
