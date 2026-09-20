#include<cuda_runtime.h>
#include <chrono>
#include<iostream>
#include<stdint.h>
#include<stdlib.h>
#include"gpu_renderer.h"


#define CUDA_ERR(x)\
do\
{ \
cudaError_t err= (x);\
if(err != cudaSuccess)\
{\
fprintf(stderr,"%s:%d : %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
exit(EXIT_FAILURE); \
}\
} \
while(0)

__device__ void point_plot_2D(u32 tidx,u32 tidy,float x,float y,cuda_state cs)
{
    int xr = roundf(cs.w/2 + x);
    int yr = roundf(cs.h/2 - y);
    if((tidx < (xr + 2) && tidx  > (xr - 2))  && (tidy < (yr + 2) && tidy  > (yr - 2)))
        cs.d_pixels[tidy * cs.w + tidx] = 0x000000FFu;
}

__device__ void point_plot_3D(u32 tidx,u32 tidy,float x,float y,float z,cuda_state cs,float z_translation= 0,int scale=100)
{
    float px = x/(z + z_translation);
    float py = y/(z + z_translation);
    // xn = w/2 - xp
    int xr = roundf(cs.w/2 + px * scale);
    int yr = roundf(cs.h/2 - py * scale);
    if((tidx < (xr + 2) && tidx  > (xr - 2))  && (tidy < (yr + 2) && tidy  > (yr - 2)))
        cs.d_pixels[tidy * cs.w + tidx] = 0x000000FFu;
}

__device__ float3 rotate_y(u32 tidx,u32 tidy,float x ,float y, float z , float t)
{
    float xr =  x * cosf(t) + z * sinf(t);
    float zr = -x * sinf(t) + z * cosf(t);
    float yr = y;
    return make_float3(xr,yr,zr);
}

__device__ float3 rotate_x(u32 tidx,u32 tidy,float x ,float y, float z , float t)
{
    float yr =  y * cosf(t) + z * sinf(t);
    float zr = -y * sinf(t) + z * cosf(t);
    float xr = x;
    return make_float3(xr,yr,zr);
}

__device__ void edge_plot_DDA(u32 tidx,u32 tidy,float x1,float y1,float z1,float x2,float y2,float z2,cuda_state cs,float z_translation= 0,int scale=100)
{
    float px1 = x1/(z1 + z_translation);
    float py1 = y1/(z1 + z_translation);
    float px2 = x2/(z2 + z_translation);
    float py2 = y2/(z2 + z_translation);
    float dx = (px2 - px1)*scale;
    float dy = (py2 - py1)*scale;

    float step = fmaxf(fabsf(dx),fabsf(dy));
    float x_inc = dx/step;
    float y_inc = dy/step;

    float real_x=px1*scale,real_y=py1*scale;
    for(int i =0 ; i < step ; i++)
    {
        real_x+=x_inc;
        real_y+=y_inc;
        point_plot_2D(tidx,tidy,real_x,real_y,cs);
    }
}

__device__ void blue_grad(u32 tidx,u32 tidy,cuda_state cs)
{
    u32 pixel_thread = tidy * cs.w + tidx;
    u32 base = 0x00000000;
    cs.d_pixels[pixel_thread] = base + tidx + tidy;
}

__device__ void grad(u32 tidx,u32 tidy,cuda_state cs)
{
    u32 pixel_thread = tidy * cs.w + tidx;
    int r = tidx * 255 / (cs.w - 1);
    int g = tidy * 255 / (cs.h - 1);
    int b = (tidx + tidy) * 255 / ((cs.w - 1) + (cs.h - 1));
    u32 base = (r << 16) | (g<<8) | b;
    cs.d_pixels[pixel_thread] = base;
}

__global__ void cuber(cuda_state cs,float theta = 0) // pass w*d
{
    u32 tidx = blockDim.x * blockIdx.x + threadIdx.x;
    u32 tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx >= cs.w || tidy >= cs.h)
        return;
    cs.d_pixels[tidy * cs.w + tidx] = 0x00000000;
    //lets try to project a cube onto a 2D plane 
    //the point plot device function converts an (x,y,z) tuple into a new coordinate system with default scale 100 pixels
    //accordingly conventionally in math we take the centre of the plane as the origin(0,0,0)
    //however we must take canonically consider a new coordinate system 
    //where origin is teh top left corner of out plane(since its a fixed plane ie out monitor)
    //going downwards from origin we get +y and towards the right +x (obviosuly there are no -ve coordinates)

    //so for our case:
    //since point_plot performs the required adjustments we can use out conventional coordinate system for arguments
    //p1 - {1,1,-1},p2 -{-1,1,-1},p3 -{1,-1,-1},p4-{-1,-1,-1},p5-{1,1,1},p6-{-1,1,1},p7-{1,-1,1},p8-{-1,-1,1}
    Vec3 vertices[8] = {
    {-8, -8, -8},
    {-8, -8,  8},
    {-8,  8, -8},
    {-8,  8,  8},
    { 8, -8, -8},
    { 8, -8,  8},
    { 8,  8, -8},
    { 8,  8,  8}
    };
    
    Vec3 array[8];

    int i =0;
    for(auto[x,y,z] : vertices)
    {
        auto [xn,yn,zn] = rotate_x(tidx,tidy,x,y,z,theta);
        array[i] = {xn,yn,zn};
        i++;
    }

    for(auto[x,y,z] : array)
    { 
        point_plot_3D(tidx,tidy,x,y,z,cs,30.0f);  
    }
    for (int i = 0; i < 8; i++)
    {
        auto [x, y, z] = vertices[i];
        auto [x1,y1,z1] = array[i];
    for (int j = i + 1; j < 8; j++)
    {
        auto [x_p, y_p, z_p] = vertices[j];
        auto [x2,y2,z2] = array[j];

        if (((x_p == x) && ((y_p == y) || (z_p == z))) ||
            ((y_p == y) && (z_p == z)))
        {
            edge_plot_DDA(
                tidx, tidy,
                x1, y1, z1,
                x2, y2, z2,
                cs, 30.0f
            );
        }
    }
    }
}

cuda_state cuda_init(int w,int h)
{
    cuda_state cs={NULL,w,h};
    u32 n = cs.w*cs.h;
    CUDA_ERR(cudaMalloc((void**)&cs.d_pixels ,sizeof(u32)*n));  
    return cs;
}

Vec2 cuda_render(cuda_state cs_gpu,u32 blockx,u32 blocky,u32 *pixels,size_t pitch,float theta)
{
    dim3 block(blockx,blocky);
    dim3 grid((cs_gpu.w + blockx -1)/ blockx ,(cs_gpu.h + blocky -1)/ blocky);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cuber<<<grid,block>>>(cs_gpu,theta);
    cudaEventRecord(stop);
    CUDA_ERR(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    CUDA_ERR(cudaGetLastError());
    auto startt = std::chrono::steady_clock::now();
    CUDA_ERR(cudaMemcpy2D(pixels,pitch ,cs_gpu.d_pixels, (cs_gpu.w)*sizeof(u32) , cs_gpu.w * sizeof(u32),cs_gpu.h,cudaMemcpyDeviceToHost));
    auto endd = std::chrono::steady_clock::now();
    float time = std::chrono::duration<float , std::milli>(endd - startt).count();
    return {ms,time};
}

void cuda_destroy(cuda_state cs)
{
    CUDA_ERR(cudaFree(cs.d_pixels));
}

