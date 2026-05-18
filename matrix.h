#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

inline void print_matrix(const std::string& name, const Matrix& M){
    std::cout<<name<<":\n";
    for(std::size_t i = 0; i < M.size(); i++){
        for(std::size_t j = 0; j < M.size(); j++){
            std::cout<<std::fixed<<std::setprecision(4)<<M[i][j]<<" ";
        }
        std::cout<<"\n";
    }
}

inline double norm(const Vector& v){
    double s = 0.0;
    for(double x : v) s += x * x;
    return std::sqrt(s);
}

inline Vector operator-(const Vector& a, const Vector& b){
    Vector res(a.size());
    for(std::size_t i = 0; i < a.size(); i++) res[i] = a[i] - b[i];
    return res;
}

inline double residual(const Matrix& A, const Vector& b, const Vector& x){
    int n = static_cast<int>(A.size());
    double res_sq = 0.0;
    for(int i = 0; i < n; i++){
        double ax = 0.0;
        for(int j = 0; j < n; j++) ax += A[i][j] * x[j];
        double r = ax - b[i];
        res_sq += r * r;
    }
    return std::sqrt(res_sq);
}

inline Matrix gen_random_matrix(int n, unsigned seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix A(n, Vector(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            A[i][j] = dist(gen);
    return A;
}

inline Vector gen_random_vector(int n, unsigned seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Vector b(n);
    for(int i = 0; i < n; i++) b[i] = dist(gen);
    return b;
}

inline Matrix gen_hilbert(int n){
    Matrix H(n, Vector(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            H[i][j] = 1.0 / (i + j + 1.0);
    return H;
}

#endif
