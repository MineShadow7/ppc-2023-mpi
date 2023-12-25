// Copyright 2023 Karagodin Andrey
#include <iostream>
#include <mpi.h>

#define SIZE 4 // размер матрицы и вектора

int main(int argc, char** argv) {
    int rank, size;
    double A[SIZE][SIZE] = {{2, -1, 0, 1}, {1, 3, -1, 1}, {0, 2, 4, -1}, {1, 2, -1, 4}};
    double b[SIZE] = {2, 1, 0, 2};
    double x[SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double x_new[SIZE];
    double eps = 1e-6;
    int max_iterations = 100;
    int iteration = 0;
    double residual;

    do {
        if (rank == 0) {
            // Выполняем одну итерацию метода Якоби на процессе 0
            for (int i = 0; i < SIZE; i++) {
                x_new[i] = b[i];
                for (int j = 0; j < SIZE; j++) {
                    if (j != i) {
                        x_new[i] -= A[i][j] * x[j];
                    }
                }
                x_new[i] /= A[i][i];
            }
        }

        // Рассылаем x_new всем процессам
        MPI_Bcast(x_new, SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вычисляем норму разницы между текущим и предыдущим приближением
        residual = 0;
        for (int i = rank; i < SIZE; i += size) {
            double diff = x_new[i] - x[i];
            residual += diff * diff;
        }
        MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residual = sqrt(residual);

        // Обновляем значения x на всех процессах для следующей итерации
        for (int i = rank; i < SIZE; i += size) {
            x[i] = x_new[i];
        }

        // Синхронизируем процессы после каждой итерации
        MPI_Barrier(MPI_COMM_WORLD);

        // Увеличиваем счетчик итераций
        iteration++;

        // Проверяем условия остановки: достигнута максимальная
        // число итераций или достигнута требуемая точность
    } while (iteration < max_iterations && residual > eps);

    // Собираем значения x со всех процессов на процессе 0
    MPI_Gather(x, SIZE / size, MPI_DOUBLE, x_new, SIZE / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Выводим результат на процессе 0
    if (rank == 0) {
        std::cout << "Solution:" << std::endl;
        for (int i = 0; i < SIZE; i++) {
            std::cout << "x[" << i << "] = " << x_new[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}