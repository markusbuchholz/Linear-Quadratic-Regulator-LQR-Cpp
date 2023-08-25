// Markus Buchholz 2023
// g++ LQR_control.cpp -o t -I/usr/include/eigen3 -I/usr/include/python3.8 -lpython3.8

// https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <numeric>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

//------------------------------------------------------------------------------------

double tolerance = 0.00001;
int max_iter = 100000;
double dt = 0.001;
int dim_x = 4;
int dim_u = 1;

//------------------------------------------------------------------------------------

void solveRiccati(Eigen::MatrixXd &A, Eigen::MatrixXd &B,
                  Eigen::MatrixXd &Q, Eigen::MatrixXd &R,
                  Eigen::MatrixXd &P, Eigen::MatrixXd &K)
{

    P = Q;

    Eigen::MatrixXd P_next;
    double diff = std::numeric_limits<double>::max();
    int ii = 0;

    while (diff > tolerance || ii > max_iter)
    {

        ii++;
        P_next = P + (A.transpose() * P + P * A + Q - P * B * R.inverse() * B.transpose() * P) * dt;
        diff = fabs((P_next - P).maxCoeff());
        P = P_next;
        // std::cout << diff << "\n";
    }

    K = R.inverse() * B.transpose() * P;
}


//------------------------------------------------------------------------------------


std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> LQR(Eigen::MatrixXd &A, Eigen::MatrixXd &B,
                                                                                                                   Eigen::MatrixXd &Q, Eigen::MatrixXd &R,
                                                                                                                   Eigen::MatrixXd &P, Eigen::MatrixXd &K)
{

    Eigen::MatrixXd Xk = Eigen::MatrixXd::Zero(dim_x, dim_u);
    Eigen::MatrixXd uk = Eigen::MatrixXd::Zero(1, 1);

    Xk << 0.0, 0.0, -0.2, 0.0;

    float dt = 0.01;

    std::vector<float> time;
    std::vector<float> xk0;
    std::vector<float> xk1;
    std::vector<float> xk2;
    std::vector<float> xk3;

    for (int ii = 0; ii < 1000; ii++)
    {

        uk = -K * Xk;
        Xk = Xk + A * Xk * dt + B * uk * dt;

        time.push_back(ii);
        xk0.push_back(Xk(0, 0));
        xk1.push_back(Xk(1, 0));
        xk2.push_back(Xk(2, 0));
        xk3.push_back(Xk(3, 0));
    }

    return std::make_tuple(time, xk0, xk1, xk2, xk3);
}

//------------------------------------------------------------------------------------

void plot(std::vector<float> time, std::vector<float> xk0, std::vector<float> xk1, std::vector<float> xk2, std::vector<float> xk3)
{
   
    plt::title("Linear Quadratic Regulation (LQR)");
    plt::named_plot("pendulum pos", time, xk0);
    //  plt::named_plot("xk1", time, xk1);
    plt::named_plot("cart pos", time, xk2, "r--");
    // plt::named_plot("xk3", time, xk3, "b--");
    plt::xlabel("time");
    plt::ylabel("Y");
    plt::legend();

    plt::show();
}

//------------------------------------------------------------------------------------

int main()
{

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim_x, dim_x);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(dim_x, dim_u);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(dim_x, dim_x);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dim_u, dim_u);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dim_x, dim_x);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(dim_u, dim_x);

    A << 0.0, 1.0, 0.0, 0.0,
        0.0, -0.1818, 2.6727, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.4545, 31.1818, 0.0;

    B << 0.0, 1.8182, 0.0, 4.5455;

    Q << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    R(0, 0) = 1.0;

    solveRiccati(A, B, Q, R, P, K);

    auto lqr = LQR(A, B, Q, R, P, K);

    plot(std::get<0>(lqr), std::get<1>(lqr), std::get<2>(lqr), std::get<3>(lqr), std::get<4>(lqr));

    return 0;
}
