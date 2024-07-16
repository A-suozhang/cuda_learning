#include <stdio.h>
#include <cstdlib>

void add_kernel(float *x, float *y, float *out, int n){
    for (int i = 0; i < n; ++i){
        out[i] = x[i] + y[i];
    }
}

int main(){
    int N = 10000000;
    size_t mem_size = sizeof(float) * N;

    float *x, *y, *out;
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    out = static_cast<float*>(malloc(mem_size));

    for(int i = 0; i < N; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    add_kernel(x, y, out, N);

    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    free(x);
    free(y);
    free(out);
}
