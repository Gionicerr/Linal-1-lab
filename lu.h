#ifndef LU_H
#define LU_H

#include "matrix.h"

#include <algorithm>
#include <cmath>

struct LU{
    Matrix L;
    Matrix U;
};

inline Vector gauss_no_pivot(Matrix A, Vector b){
    int n = static_cast<int>(A.size());
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            double f = A[j][i] / A[i][i];
            b[j] -= f * b[i];
            for(int k = i; k < n; k++) A[j][k] -= f * A[i][k];
        }
    }
    Vector x(n);
    for(int i = n - 1; i >= 0; i--){
        double s = 0.0;
        for(int j = i + 1; j < n; j++) s += A[i][j] * x[j];
        x[i] = (b[i] - s) / A[i][i];
    }
    return x;
}

inline Vector gauss_partial_pivot(Matrix A, Vector b){
    int n = static_cast<int>(A.size());
    for(int i = 0; i < n; i++){
        int pivot = i;
        for(int j = i + 1; j < n; j++)
            if(std::abs(A[j][i]) > std::abs(A[pivot][i])) pivot = j;
        std::swap(A[i], A[pivot]);
        std::swap(b[i], b[pivot]);
        for(int j = i + 1; j < n; j++){
            double f = A[j][i] / A[i][i];
            b[j] -= f * b[i];
            for(int k = i; k < n; k++) A[j][k] -= f * A[i][k];
        }
    }
    Vector x(n);
    for(int i = n - 1; i >= 0; i--){
        double s = 0.0;
        for(int j = i + 1; j < n; j++) s += A[i][j] * x[j];
        x[i] = (b[i] - s) / A[i][i];
    }
    return x;
}

inline LU lu_decompose(const Matrix& A){
    int n = static_cast<int>(A.size());
    Matrix L(n, Vector(n, 0.0)), U(n, Vector(n, 0.0));
    for(int i = 0; i < n; i++){
        for(int k = i; k < n; k++){
            double s = 0.0;
            for(int j = 0; j < i; j++) s += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - s;
        }
        for(int k = i; k < n; k++){
            if(k == i) L[i][i] = 1.0;
            else{
                double s = 0.0;
                for(int j = 0; j < i; j++) s += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - s) / U[i][i];
            }
        }
    }
    return {L, U};
}

inline Vector forward_sub(const Matrix& L, const Vector& b){
    int n = static_cast<int>(L.size());
    Vector y(n);
    for(int i = 0; i < n; i++){
        double s = 0.0;
        for(int j = 0; j < i; j++) s += L[i][j] * y[j];
        y[i] = b[i] - s;
    }
    return y;
}

inline Vector backward_sub(const Matrix& U, const Vector& y){
    int n = static_cast<int>(U.size());
    Vector x(n);
    for(int i = n - 1; i >= 0; i--){
        double s = 0.0;
        for(int j = i + 1; j < n; j++) s += U[i][j] * x[j];
        x[i] = (y[i] - s) / U[i][i];
    }
    return x;
}

#endif
