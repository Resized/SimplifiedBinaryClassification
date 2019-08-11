// InitialMPIproject.cpp : Defines the entry point for the console application.
//

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/

/* This is an interactive version of cpi */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

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

vector <float> xx;

float * weights;


int readPointFromFile(FILE * fp, POINT * points, int * pointNum, int K)
{

	if (feof(fp))
	{
		return 0;
	}
	for (int i = 0; i < K; i++)
	{
		if (!fscanf(fp, "%f ", &(points[*pointNum].x[i])))
			return 0;
	}
	points[*pointNum].x[K] = 1;
	if (!fscanf(fp, "%d \n", &(points[*pointNum].set)))
		return 0;
	(*pointNum)++;
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
	double result = 0;
	for (int i = 0; i < K + 1; i++)
	{
		result += weights[i] * x[i];
	}
	//result += weights[K];
	return result;
}

void updateWeights(float * x, float * weights, int signResult, float alpha, int K)
{
	for (int i = 0; i < K + 1; i++)
	{
		weights[i] = weights[i] + alpha*-signResult*x[i];
	}
	//weights[K] = weights[K] + alpha*signResult;
}

void zeroWeights(float * weights, int K)
{
	for (int i = 0; i < K + 1; i++)
	{
		weights[i] = 0;
	}
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
	int N, Nmis = 0, K, LIMIT, iterCounter = 0, pointNum = 0;
	bool isClassifiedProperly = false;
	bool isAlphaFound = false;
	double t1, t2;
	POINT * points;
	FILE * fp;

	fp = fopen("data1.txt", "r");
	if (fp == NULL)
	{
		printf("file could not be opened\n");
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	t1 = MPI_Wtime();

	// N    K    alphaZero   alphaMax LIMIT   QC
	fscanf(fp, "%d %d %f %f %d %f \n", &N, &K, &alphaZero, &alphaMax, &LIMIT, &QC);
	points = (POINT*)malloc(sizeof(POINT) * N);
	weights = (float*)calloc(K + 1, sizeof(float));

	for (int i = 0; i < N; i++)
	{
		points[i].x = (float*)malloc(sizeof(double) * (K + 1));
	}

	for (int i = 0; i < N; i++)
	{
		readPointFromFile(fp, points, &pointNum, K);
	}

	fclose(fp);

	alpha = alphaZero;
	int signResult = 0;

	while (true)
	{
		zeroWeights(weights, K);
		while (!isClassifiedProperly && iterCounter < LIMIT)
		{
			iterCounter++;
			isClassifiedProperly = true;
			for (int i = 0; i < N; i++)
			{
				signResult = sign(func(points[i].x, weights, K));
				if (signResult != points[i].set)
				{
					updateWeights(points[i].x, weights, signResult, alpha, K);
					isClassifiedProperly = false;
					break;
				}
			}
		}
		Nmis = 0;
		for (int i = 0; i < N; i++)
		{
			signResult = sign(func(points[i].x, weights, K));
			if (signResult != points[i].set)
			{
				Nmis++;
			}
		}
		quality = (float)Nmis / N;

		if (quality < QC)
		{
			break;
		}
		alpha += alphaZero;
		if (alpha > alphaMax)
		{
			break;
		}
	}

	t2 = MPI_Wtime();

	printf("minimal alpha value: %1.4f\n", alpha);
	printf("quality value: %1.5f\n", quality);
	printf("weight 0: %1.4f\n", weights[0]);
	printf("weight 1: %1.4f\n", weights[1]);
	printf("weight 2: %1.4f\n", weights[2]);

	printf("MPI_Wtime: %1.4f\n", t2 - t1);// fflush(stdout);

	// free memory
	for (int i = 0; i < N; i++)
	{
		free(points[i].x);
	}
	free(points);
	free(weights);


	//int x = 7, y = 10, answer = 77777;
	//if (myid == 0) {
	//	MPI_Send(&x, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	//	MPI_Recv(&answer, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
	//}
	//else {
	//	MPI_Recv(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	//	y = y * 3;
	//	MPI_Send(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	//}

	//if (myid == 0)
	//printf("answer = %d numprocs = %d  myid = %d   %s\n", answer, numprocs, myid, processor_name);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
	return 0;
}

