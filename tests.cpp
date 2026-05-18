#include "tests.h"

#include "lu.h"
#include "matrix.h"

#include <chrono>
#include <iomanip>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

double ms(Clock::time_point t){
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count() / 1000.0;
}

void test_1(){
    std::cout<<"\nTime copmapre\n";
    std::cout<<std::setw(5)<<"n";
    std::cout<<std::setw(18)<<"Gauss(no pivot)";
    std::cout<<std::setw(18)<<"Gauss(pivot)";
    std::cout<<std::setw(18)<<"LU total";
    std::cout<<std::setw(18)<<"LU decomp";
    std::cout<<std::setw(18)<<"LU solve\n";
    std::cout<<std::string(95, '-')<<"\n";

    for(int n : {100, 200, 500, 1000}){
        Matrix A = gen_random_matrix(n, 42);
        Vector b = gen_random_vector(n, 43);

        auto t0 = Clock::now();
        gauss_no_pivot(A, b);
        double t_no = ms(t0);
        auto t1 = Clock::now();
        gauss_partial_pivot(A, b);
        double t_piv = ms(t1);
        auto t2 = Clock::now();
        auto [L, U] = lu_decompose(A);
        double t_dec = ms(t2);
        auto t3 = Clock::now();
        Vector y = forward_sub(L, b);
        Vector x = backward_sub(U, y);
        double t_sol = ms(t3);

        std::cout<<std::setw(5)<<n;
        std::cout<<std::setw(18)<<std::fixed<<std::setprecision(3)<<t_no;
        std::cout<<std::setw(18)<<t_piv;
        std::cout<<std::setw(18)<<(t_dec + t_sol);
        std::cout<<std::setw(18)<<t_dec;
        std::cout<<std::setw(18)<<t_sol<<"\n";
    }
}

void test_2(){
    std::cout<<"\nTime economy\n";
    std::cout<<std::setw(5)<<"k";
    std::cout<<std::setw(20)<<"Gauss(pivot) total";
    std::cout<<std::setw(20)<<"LU total\n";
    std::cout<<std::string(50, '-')<<"\n";

    int n = 500;
    Matrix A = gen_random_matrix(n, 42);
    auto [L, U] = lu_decompose(A);

    for(int k : {1, 10, 100}){
        auto t0 = Clock::now();
        for(int i = 0; i < k; i++){
            Vector b = gen_random_vector(n, 44 + i);
            gauss_partial_pivot(A, b);
        }
        double t_gauss = ms(t0);

        auto t1 = Clock::now();
        for(int i = 0; i < k; i++){
            Vector b = gen_random_vector(n, 44 + i);
            Vector y = forward_sub(L, b);
            Vector x = backward_sub(U, y);
        }
        double t_lu = ms(t1);

        std::cout<<std::setw(5)<<k;
        std::cout<<std::setw(20)<<std::fixed<<std::setprecision(3)<<t_gauss;
        std::cout<<std::setw(20)<<t_lu<<"\n";
    }
}

void test_3(){
    std::cout<<"\nAccuracy on Hilbert matrices\n";
    std::cout<<std::setw(5)<<"n";
    std::cout<<std::setw(22)<<"Rel.Err (NoPiv)";
    std::cout<<std::setw(22)<<"Resid (NoPiv)";
    std::cout<<std::setw(22)<<"Rel.Err (Piv)";
    std::cout<<std::setw(22)<<"Resid (Piv)\n";
    std::cout<<std::string(95, '-')<<"\n";

    for(int n : {5, 10, 15}){
        Matrix H = gen_hilbert(n);
        Vector x_true(n, 1.0);
        Vector b(n, 0.0);
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++) b[i] += H[i][j] * x_true[j];

        auto x_no = gauss_no_pivot(H, b);
        auto x_piv = gauss_partial_pivot(H, b);

        double err_no = norm(x_no - x_true) / norm(x_true);
        double err_piv = norm(x_piv - x_true) / norm(x_true);
        double res_no = residual(H, b, x_no);
        double res_piv = residual(H, b, x_piv);

        std::cout<<std::setw(5)<<n;
        std::cout<<std::setw(22)<<std::scientific<<std::setprecision(6)<<err_no;
        std::cout<<std::setw(22)<<res_no;
        std::cout<<std::setw(22)<<err_piv;
        std::cout<<std::setw(22)<<res_piv<<"\n";
    }
}
