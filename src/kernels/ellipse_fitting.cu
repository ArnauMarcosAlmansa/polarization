#include <cuda.h>
#include <math.h>

__device__ float clamp_grad(float grad)
{
    // Define the clamp_grad function as per your need
    // Placeholder implementation:
    return std::max(std::min(grad, 0.25f), -0.25f);
}

__device__ bool isclose(float a, float b, float abs_tol)
{
    return fabs(a - b) <= std::max(1e-05 * std::max(fabs(a), fabs(b)), abs_tol);
}

__device__ float diff_a(float a, float b, float A, float x, float y)
{
    return -4 * pow((x * cos(A) + y * sin(A)), 2) * (pow((x * cos(A) + y * sin(A)), 2) / pow(a, 2) + pow((y * cos(A) - x * sin(A)), 2) / pow(b, 2) - 1) /
           pow(a, 3);
}

__device__ float diff_b(float a, float b, float A, float x, float y)
{
    return -4 * pow(y * cos(A) - x * sin(A), 2) * (pow(x * cos(A) + y * sin(A), 2) / pow(a, 2) + pow(y * cos(A) - x * sin(A), 2) / pow(b, 2) - 1) /
           pow(b, 3);
}

__device__ float diff_A(float a, float b, float A, float x, float y)
{
    return 4 * (pow(x * cos(A) + y * sin(A), 2) / pow(a, 2) + pow(y * cos(A) - x * sin(A), 2) / pow(b, 2) - 1) * ((x * cos(A) + y * sin(A)) * (y * cos(A) - x * sin(A)) / pow(a, 2) - (x * cos(A) + y * sin(A)) * (y * cos(A) - x * sin(A)) / pow(b, 2));
}

__global__ void kernel_fit_ellipses_using_gd(float *I0, float *I45, float *I90, float *I135, float *results,
                                             int width, int height, int max_iters = 100000,
                                             float tolerance = 0.00001, float black_threshold = 1.0 / 255)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;

    float i0 = I0[idx];
    float i45 = I45[idx];
    float i90 = I90[idx];
    float i135 = I135[idx];

    float s0 = (i0 + i45 + i90 + i135) / 2;
    float s1 = (i0 - i90);
    float s2 = (i45 - i135);

    if (s0 < black_threshold)
    {
        results[4 * idx + 0] = 1.0 / 512;
        results[4 * idx + 1] = 1.0 / 512;
        results[4 * idx + 2] = 0;
        return;
    }

    float dolp = sqrt(s1 * s1 + s2 * s2) / s0 / 2;

    float points[8][2];

    points[0][0] = 0.0f;
    points[0][1] = i0;
    points[1][0] = 0.0f;
    points[1][1] = -i0;
    points[2][0] = i45 * sin(M_PI / 4);
    points[2][1] = i45 * cos(M_PI / 4);
    points[3][0] = -i45 * sin(M_PI / 4);
    points[3][1] = -i45 * cos(M_PI / 4);
    points[4][0] = i90;
    points[4][1] = 0.0f;
    points[5][0] = -i90;
    points[5][1] = 0.0f;
    points[6][0] = i135 * sin(M_PI / 4);
    points[6][1] = -i135 * cos(M_PI / 4);
    points[7][0] = -i135 * sin(M_PI / 4);
    points[7][1] = i135 * cos(M_PI / 4);

    float a_candidate = 0;
    float b_candidate = 9999999;

    for (int i = 0; i < 8; i++)
    {
        a_candidate = std::max(a_candidate, sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1]));
        b_candidate = std::min(b_candidate, sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1]));
    }

    float a = b_candidate;
    float b = b_candidate;
    float A = 0.5 * atan2(s2, s1);

    float lr = 0.005;

    float lmbda_a = 0.1;
    float lmbda_b = 0.1;

    float lambda_ridge = 1 - dolp;
    float lambda_lasso = dolp;

    bool converged = true;
    for (int i = 0; i < max_iters; i++)
    {
        float delta_a = 0;
        float delta_b = 0;
        float delta_A = 0;

        for (int j = 0; j < 8; j++)
        {
            float *point = points[j];
            float grad_a = diff_a(a, b, A, point[0], point[1]);
            float grad_b = diff_b(a, b, A, point[0], point[1]);
            float grad_A = diff_A(a, b, A, point[0], point[1]);

            delta_a += clamp_grad(grad_a);
            delta_b += clamp_grad(grad_b);
            delta_A += clamp_grad(grad_A);
        }

        float reg_a = lmbda_a * (lambda_lasso * a / fabs(a) + lambda_ridge * 2 * a);
        float reg_b = lmbda_b * (lambda_lasso * b / fabs(b) + lambda_ridge * 2 * b);

        float new_a = a - ((delta_a / 8 + reg_a) * lr);
        float new_b = b - ((delta_b / 8 + reg_b) * lr);
        float new_A = A - ((delta_A / 8) * lr);

        if (isclose(a, new_a, tolerance) && isclose(b, new_b, tolerance) && isclose(A, new_A, tolerance))
        {
            break;
        }

        a = new_a;
        b = new_b;
        A = new_A;
    }

    results[4 * idx + 0] = a;
    results[4 * idx + 1] = b;
    results[4 * idx + 2] = A;
}
