#ifndef gpu_render
#define gpu_render
#include<stdint.h>
typedef uint32_t u32;
typedef struct Vec3
{
    float x;
    float y;
    float z;
}Vec3;

struct Vec2 {
    float x;
    float y;
};

typedef struct cuda_state
{
    u32 *d_pixels;
    int w;
    int h;
}cuda_state;
cuda_state cuda_init(int w, int h);
Vec2 cuda_render(cuda_state cs_gpu,u32 blockx,u32 blocky,u32 *pixels,size_t pitch,float theta=0);
void cuda_destroy(cuda_state cs_gpu);

#endif
