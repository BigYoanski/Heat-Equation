#include "utils_1d.h"


/* Implement the MPI solver here. You may add includes and helper functions.
 * Keep the explicit finite difference scheme and the public signature.
 * MPI is initialized/finalized by the driver. The full result and cumulative
 * error must be available on rank 0 when this function returns.
 */

 /**
 * @brief Solve the heat equation using an explicit finite difference scheme.
 *
 * This functions solves the heat equation iterating over ITER time steps.
 *
 * @param heat_array Array of `double` values representing the temperatures of the points.
 * @param err Pointer to a `double` value representing the error in the simulation.
 * @param num_pts Number of points in the simulation.
 *
 * @return 0 on success, -1 on error.
 */
int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Setup the parameters
  double dx = 1.0/(num_pts+1);
  double dt = 0.5*dx*dx;
  size_t beg = 1;
  double local_err = 0.0;

  //uneven distribution of num_pts among the processes
  int base = num_pts / size;
  int remainder = num_pts % size;
  int local_n = base;
  if (rank < remainder) {
      local_n++;
  }
  size_t end = beg + local_n - 1;

  int* displs = malloc(size * sizeof(int));
  int* sendcounts = malloc(size * sizeof(int));

  for (int i = 0; i < size; i++) {
      sendcounts[i] = base;
      if (i < remainder) {
          sendcounts[i]++;
      }
  }

  displs[0] = 0;

  for (int i = 1; i < size; i++) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
  }

  int start_local_arr = displs[rank] + 1; //where each rank's portion of the array starts

  // Utilize an auxiliary array to store the new temperatures
  double* aux = (double*)malloc((local_n + 2) * sizeof(double));
  double* local_arr = (double*)malloc((local_n + 2) * sizeof(double));

  aux[beg - 1] = 0.0;
  local_arr[end + 1] = 0.0;
  local_arr[beg - 1] = 0.0;
  local_arr[end + 1] = 0.0;


  //each rank's portion gets saved
  for (int i = beg; i <= local_n; i++) {
      local_arr[i] = heat_array[start_local_arr + i - 1];
  }
  

  for (size_t i = 0; i < ITER; i++) {
      MPI_Request reqs[4];
      int count = 0;

      if (rank > 0) {
          MPI_Isend(&local_arr[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[count]);
          count++;
      }

      if (rank < size - 1) {
          MPI_Isend(&local_arr[end], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[count]);
          count++;
      }

      if (rank > 0) {
          MPI_Irecv(&local_arr[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[count]);
          count++;
      }

      if (rank < size - 1) {
          MPI_Irecv(&local_arr[end + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[count]);
          count++;
      }

      //calculating the internal points while the outer ones are being sent
      for (size_t j = beg + 1; j < end; j++) {
          aux[j] = local_arr[j] + dt / (dx * dx) * (local_arr[j - 1] - 2.0 * local_arr[j] + local_arr[j + 1]);
          local_err += dx * fabs(aux[j] - local_arr[j]);
      }

      MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);

      aux[end] = local_arr[end]
         + dt / (dx * dx) *
           (local_arr[end - 1] - 2.0 * local_arr[end]
            + local_arr[end + 1]);

      local_err += dx * fabs(aux[end] - local_arr[end]);


      //swap
      double* temp = local_arr;
      local_arr = aux;
      aux = temp;

  }
  
  MPI_Reduce(&local_err, err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Gatherv(&local_arr[1], local_n, MPI_DOUBLE, &heat_array[1], sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(displs);
  free(sendcounts);
  free(aux);
  free(local_arr);
  return 0;
}
