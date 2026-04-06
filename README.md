# Linal-1-lab
Первая лаба по линалу


## Постановка задачи
Реализовать набор функций для решения систем линейных алгебраических уравнений (СЛАУ) методом Гаусса и методом LU-разложения.
Провести сравнительный анализ времени работы алгоритмов и продемонстрировать преимущество LU-разложения при решении систем с одинаковой матрицей, но разными правыми частями.

## Краткое описание реализованных алгоритмов
LU-разложение (LU-декомпозиция, LU-факторизация) — представление матрицы А в виде произведения двух матриц, *A = LU*, где *L* - нижняя треугольная матрица, а *U* - верхняя треугольная матрица.

Ме́тод Га́усса — классический метод решения системы ленейных алгебраических уравнений (СЛАУ). Это метод последовательного исключения переменных, когда с помощью элементарных преобразований система уравнений приводится к равносильной системе правого верхнего (традиционного), правого нижнего, левого верхнего или левого нижнего треугольного вида, из которой последовательно, начиная с последних или c первых (по номеру), находятся все переменные системы.

Матрицей Гильберта называется квадратная матрица *H* с эклкментами $H_ij$ = $\frac{1}{i + j -1}$,*i*,*j* = 1,2,3,...,n.

## Описание структуры программы

```
double norm(const Vector& v){
    double s = 0.0;
    for(double x : v) s += x * x;
    return std::sqrt(s);
}

Vector operator-(const Vector& a, const Vector& b){
    Vector res(a.size());
    for(size_t i = 0; i < a.size(); i++) res[i] = a[i] - b[i];
    return res;
}

double residual(const Matrix& A, const Vector& b, const Vector& x){
    int n = A.size();
    double res_sq = 0.0;
    for(int i = 0; i < n; i++){
        double ax = 0.0;
        for(int j = 0; j < n; j++) ax += A[i][j] * x[j];
        double r = ax - b[i];
        res_sq += r * r;
    }
    return std::sqrt(res_sq);
}
```
Вычисление нормы для расчета относительной погрещности, получение вектора разностей для подстановки в norm, вычисление невязки.

```
Matrix gen_random_matrix(int n, unsigned seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix A(n, Vector(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            A[i][j] = dist(gen);
    return A;
}

Vector gen_random_vector(int n, unsigned seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Vector b(n);
    for(int i = 0; i < n; i++) b[i] = dist(gen);
    return b;
}

Matrix gen_hilbert(int n){
    Matrix H(n, Vector(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            H[i][j] = 1.0 / (i + j + 1.0);
    return H;
}
```
Создает данные для тестов с фикс. сидом.

```
Vector gauss_no_pivot(Matrix A, Vector b){
    int n = A.size();
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

Vector gauss_partial_pivot(Matrix A, Vector b){
    int n = A.size();
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

struct LU{ Matrix L; Matrix U; };
LU lu_decompose(const Matrix& A){
    int n = A.size();
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

Vector forward_sub(const Matrix& L, const Vector& b){
    int n = L.size();
    Vector y(n);
    for(int i = 0; i < n; i++){
        double s = 0.0;
        for(int j = 0; j < i; j++) s += L[i][j] * y[j];
        y[i] = b[i] - s;
    }
    return y;
}

Vector backward_sub(const Matrix& U, const Vector& y){
    int n = U.size();
    Vector x(n);
    for(int i = n - 1; i >= 0; i--){
        double s = 0.0;
        for(int j = i + 1; j < n; j++) s += U[i][j] * x[j];
        x[i] = (y[i] - s) / U[i][i];
    }
    return x;
}
```
Два Гаусса, LU с двумя ходами.

```
using Clock = std::chrono::high_resolution_clock;
double ms(Clock::time_point t){
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count() / 1000.0;
}
```
Для времени.

```
void test_1(){
    std::cout<<"\n=== 4.1 Сравнение времени решения одной системы ===\n";
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
    std::cout<<"\nЭкономия времени при множественных правых частях\n";
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
    std::cout<<"\nТочность на матрицах Гильберта\n";
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
```
Тесты.

```
int main(){
    Matrix testA = {{2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {6.0, 7.0, 8.0}};
    std::cout<<"Исходная матрица A:\n";
    for(int i = 0; i < testA.size(); i++){
        for(int j = 0; j < testA.size(); j++)
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
```
Мейн. (В нем проверил работу LU потому что казалось что чтото не так).

## Результаты экспериментов

