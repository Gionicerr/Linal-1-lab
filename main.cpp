#include "lu.h"
#include "matrix.h"
#include "tests.h"

#include <iostream>

int main(){
    Matrix testA = {{2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {6.0, 7.0, 8.0}};
    std::cout<<"Matrix A:\n";
    for(size_t i = 0; i < testA.size(); i++){
        for(size_t j = 0; j < testA.size(); j++)
            std::cout<<testA[i][j]<<" ";
        std::cout<<"\n";
    }

    auto lu = lu_decompose(testA);
    print_matrix("L", lu.L);
    print_matrix("U", lu.U);
    test_1();
    test_2();
    test_3();
    return 0;
}
