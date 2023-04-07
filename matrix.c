#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, double low, double high) {
    srand(42);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
	if (rows < 1 || cols < 1) {
		PyErr_SetString(PyExc_TypeError, "ERROR: The given dimensions are invalid!");
		return -1;
	}
	
	*mat = malloc(sizeof(matrix));
	
	if (*mat == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "ERROR: Matrix memory allocation failed!");
		return -1;
	}
	
	(*mat)->data = malloc(rows * cols * sizeof(double));

	if ((*mat)->data == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "ERROR: Matrix data memory allocation failed!");
		free(*mat);
		return -1;
	}

	(*mat)->rows = rows;
	(*mat)->cols = cols;

	fill_matrix(*mat, 0.0);

	(*mat)->parent = NULL;
	(*mat)->ref_cnt = 1;

	return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
	if (rows < 1 || cols < 1) {
		PyErr_SetString(PyExc_TypeError, "ERROR: The given dimensions are invalid!");
		return -1;
	}

	*mat = malloc(sizeof(matrix));
	
	if (*mat == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "ERROR: Matrix memory allocation failed!");
		return -1;
	}

	(*mat)->rows = rows;
	(*mat)->cols = cols;
	
	(*mat)->data = (from->data) + offset;
	(*mat)->parent = from;
	(*mat)->ref_cnt = ((*mat)->parent->ref_cnt) + 1;
	(*mat)->parent->ref_cnt = ((*mat)->ref_cnt);

	return 0;
}

/*
 * This function frees the matrix struct pointed to by `mat`. However, you need to make sure that
 * you only free the data if `mat` is not a slice and has no existing slices, or if `mat` is the
 * last existing slice of its parent matrix and its parent matrix has no other references.
 * You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
	if (mat) {
		if (mat->parent) {
			mat->ref_cnt = mat->parent->ref_cnt;
			mat->parent->ref_cnt -= 1;
		} else {
			mat->ref_cnt -= 1;
		}

		if (mat->ref_cnt <= 0) {
			free(mat->data);
		}
	}

	if (mat && mat->parent) {
		if (mat->parent->ref_cnt <= 0) {
			deallocate_matrix(mat->parent);
		}

		free(mat);
	} else if (mat && mat->parent == NULL && mat->ref_cnt <= 0) {
		free(mat);
	}
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
	return mat->data[(row * mat->cols) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
	mat->data[(row * mat->cols) + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
	/*
	#pragma omp parallel for
	for (int i = 0; i < size / 4 * 4; i+= 4) {
		matdata[i] = val;
		matdata[i + 1] = val;
		matdata[i + 2] = val;
		matdata[i + 3] = val;
	}

	#pragma omp parallel for
	for (int i = size / 4 * 4; i < size; i++) {
		matdata[i] = val;
	}
	*/

	// SIMD Version
	__m256d valueVector = _mm256_set1_pd(val);
	double *matdata = mat->data;
	int size = (mat->rows * mat->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i+= 16) {
		_mm256_storeu_pd(matdata + i, valueVector);
		_mm256_storeu_pd(matdata + (i + 4), valueVector);
		_mm256_storeu_pd(matdata + (i + 8), valueVector);
		_mm256_storeu_pd(matdata + (i + 12), valueVector);
	}

	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		matdata[i] = val;
	}
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
	/*
	double *resdata = result->data;
	double *mat1data = mat1->data;
	double *mat2data = mat2->data;
	int size = (result->rows * result->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 4 * 4; i += 4) {
		resdata[i] = mat1data[i] + mat2data[i];
		resdata[i + 1] = mat1data[i + 1] + mat2data[i + 1];
		resdata[i + 2] = mat1data[i + 2] + mat2data[i + 2];
		resdata[i + 3] = mat1data[i + 3] + mat2data[i + 3];
	}

	#pragma omp parallel for
	for (int i = size / 4 * 4; i < size; i++) {
		resdata[i] = mat1data[i] + mat2data[i];
	}
	*/
	
	// SIMD Version
	__m256d mat1Vector;
	__m256d mat2Vector;
	double *resdata = result->data;
	double *mat1data = mat1->data;
	double *mat2data = mat2->data;
	int size = (result->rows * result->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i += 16) {
		mat1Vector = _mm256_loadu_pd(mat1data + i);
		mat2Vector = _mm256_loadu_pd(mat2data + i);
		_mm256_storeu_pd(resdata + i, _mm256_add_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 4));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 4));
		_mm256_storeu_pd(resdata + (i + 4), _mm256_add_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 8));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 8));
		_mm256_storeu_pd(resdata + (i + 8), _mm256_add_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 12));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 12));
		_mm256_storeu_pd(resdata + (i + 12), _mm256_add_pd(mat1Vector, mat2Vector));
	}

	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		resdata[i] = mat1data[i] + mat2data[i];
	}

	return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
	/*
	double *resdata = result->data;
	double *mat1data = mat1->data;
	double *mat2data = mat2->data;
	int size = (result->rows * result->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 4 * 4; i += 4) {
		resdata[i] = mat1data[i] - mat2data[i];
		resdata[i + 1] = mat1data[i + 1] - mat2data[i + 1];
		resdata[i + 2] = mat1data[i + 2] - mat2data[i + 2];
		resdata[i + 3] = mat1data[i + 3] - mat2data[i + 3];
	}

	#pragma omp parallel for
	for (int i = size / 4 * 4; i < size; i++) {
		resdata[i] = mat1data[i] - mat2data[i];
	}
	*/
	
	// SIMD Version
	__m256d mat1Vector;
	__m256d mat2Vector;
	double *resdata = result->data;
	double *mat1data = mat1->data;
	double *mat2data = mat2->data;
	int size = (result->rows * result->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i += 16) {
		mat1Vector = _mm256_loadu_pd(mat1data + i);
		mat2Vector = _mm256_loadu_pd(mat2data + i);
		_mm256_storeu_pd(resdata + i, _mm256_sub_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 4));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 4));
		_mm256_storeu_pd(resdata + (i + 4), _mm256_sub_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 8));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 8));
		_mm256_storeu_pd(resdata + (i + 8), _mm256_sub_pd(mat1Vector, mat2Vector));
		mat1Vector = _mm256_loadu_pd(mat1data + (i + 12));
		mat2Vector = _mm256_loadu_pd(mat2data + (i + 12));
		_mm256_storeu_pd(resdata + (i + 12), _mm256_sub_pd(mat1Vector, mat2Vector));
	}

	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		resdata[i] = mat1data[i] - mat2data[i];
	}

	return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
	fill_matrix(result, 0.0);
	int i, j, k;

	double *resdata = result->data;
	double *mat1data = mat1->data;
	double *mat2data = mat2->data;
	int mat1rows = mat1->rows;
	int mat1cols = mat1->cols;
	int mat2cols = mat2->cols;
	int rescols = result->cols;

	for (i = 0; i < mat1rows; i++) {
		for (k = 0; k < mat1cols; k++){
			for (j = 0; j < mat2cols / 4 * 4; j += 4) {
				resdata[j + i * rescols] += (mat1data[k + i * mat1cols]) * (mat2data[j + k * mat2cols]);
				resdata[j + 1 + i * rescols] += (mat1data[k + i * mat1cols]) * (mat2data[j + 1 + k * mat2cols]);
				resdata[j + 2 + i * rescols] += (mat1data[k + i * mat1cols]) * (mat2data[j + 2 + k * mat2cols]);
				resdata[j + 3 + i * rescols] += (mat1data[k + i * mat1cols]) * (mat2data[j + 3 + k * mat2cols]);
			}

			for (j = mat2cols / 4 * 4; j < mat2cols; j++) {
				resdata[j + i * rescols] += (mat1data[k + i * mat1cols]) * (mat2data[j + k * mat2cols]);
			}
		}
	}

	return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
	// Repeated Squaring iterative implementation

	matrix zero;
	matrix *zero_ptr1 = &zero;
	matrix **zero_ptr2 = &zero_ptr1;
	allocate_matrix(zero_ptr2, mat->rows, mat->cols);
	
	matrix x;
	matrix *x_ptr1 = &x;
	matrix **x_ptr2 = &x_ptr1;
	allocate_matrix(x_ptr2, mat->rows, mat->cols);

	matrix x_copy;
	matrix *x_copy_ptr1 = &x_copy;
	matrix **x_copy_ptr2 = &x_copy_ptr1;
	allocate_matrix(x_copy_ptr2, mat->rows, mat->cols);

	matrix res_copy;
	matrix *res_copy_ptr1 = &res_copy;
	matrix **res_copy_ptr2 = &res_copy_ptr1;
	allocate_matrix(res_copy_ptr2, mat->rows, mat->cols);

	double *resdata = result->data;
	int resrows = result->rows;
	int rescols = result->cols;
	int matcols = mat->cols;

	for (int r = 0; r < resrows; r++) {
		for (int c = 0; c < rescols; c++) {
			if (r == c) {
				resdata[(r * matcols) + c] = 1.0;
			} else {
				resdata[(r * matcols) + c] = 0.0;
			}
		}
	}

	add_matrix(x_ptr1, res_copy_ptr1, mat);
	add_matrix(x_copy_ptr1, res_copy_ptr1, mat);
	add_matrix(res_copy_ptr1, res_copy_ptr1, result);

	while (pow != 0) {
		if (pow % 2 == 1) {
			mul_matrix(result, x_ptr1, res_copy_ptr1);
			add_matrix(res_copy_ptr1, zero_ptr1, result);
			pow -= 1;

			if (pow == 0) {
				break;
			}
		}

		mul_matrix(x_ptr1, x_copy_ptr1, x_copy_ptr1);
		add_matrix(x_copy_ptr1, zero_ptr1, x_ptr1);
		pow /= 2;
	}

	deallocate_matrix(zero_ptr1);
	deallocate_matrix(res_copy_ptr1);
	deallocate_matrix(x_copy_ptr1);
	deallocate_matrix(x_ptr1);
	
	return 0;

}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
	/*
	double *resdata = result->data;
	double *matdata = mat->data;
	int size = (mat->rows * mat->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 4 * 4; i += 4) {
		resdata[i] = matdata[i] * (-1.0);
		resdata[i + 1] = matdata[i + 1] * (-1.0);
		resdata[i + 2] = matdata[i + 2] * (-1.0);
		resdata[i + 3] = matdata[i + 3] * (-1.0);
	}

	#pragma omp parallel for
	for (int i = size / 4 * 4; i < size; i++) {
		resdata[i] = matdata[i] * (-1.0);
	}
	*/

	// SIMD Version
	__m256d matVector;
	__m256d negVector = _mm256_set1_pd(-1.0);
	double *resdata = result->data;
	double *matdata = mat->data;
	int size = (mat->rows * mat->cols);

	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i += 16) {
		matVector = _mm256_loadu_pd(matdata + i);
		_mm256_storeu_pd(resdata + i, _mm256_mul_pd(matVector, negVector));
		matVector = _mm256_loadu_pd(matdata + (i + 4));
		_mm256_storeu_pd(resdata + (i + 4), _mm256_mul_pd(matVector, negVector));
		matVector = _mm256_loadu_pd(matdata + (i + 8));
		_mm256_storeu_pd(resdata + (i + 8), _mm256_mul_pd(matVector, negVector));
		matVector = _mm256_loadu_pd(matdata + (i + 12));
		_mm256_storeu_pd(resdata + (i + 12), _mm256_mul_pd(matVector, negVector));
	}

	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		resdata[i] = matdata[i] * (-1.0);
	}

	return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
	double *resdata = result->data;
	double *matdata = mat->data;
	int size = (mat->rows * mat->cols);

	#pragma omp parallel for
    for (int i = 0; i < size; i++) {
    	if (matdata[i] < 0) {
    		resdata[i] = matdata[i] * (-1.0);
    	} else {
    		resdata[i] = matdata[i];
    	}
	}

	return 0;
}