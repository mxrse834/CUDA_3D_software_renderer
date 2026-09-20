#include<iostream>
#include"gpu_renderer.h"
#include<chrono>
#include<SDL3/SDL_hints.h>
#include<SDL3/SDL_init.h>
#include<SDL3/SDL_error.h>
#include<SDL3/SDL_events.h>
#include<SDL3/SDL_video.h>
// #include<SDL3/SDL_render.h>
#include<SDL3/SDL_pixels.h>
#include<SDL3/SDL_hints.h>
#include<SDL3/SDL_surface.h>
#include<stdint.h>
#include<stdlib.h>


typedef uint32_t u32;

//ERROR macro
#define check_err(msg) \
do \
{\
    const char * err = SDL_GetError(); \
    fprintf(stderr , "%s:%d: %s Error - %s\n",__FILE__,__LINE__,msg,err);\
    SDL_ClearError();\
    exit(EXIT_FAILURE);\
}while(0)



int main()
{
    //INIT SDL
    if (!SDL_SetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION, "0"))
    check_err("SDL_SetHint");
    bool init_stat = SDL_Init(SDL_INIT_VIDEO);
    if(!init_stat)
    check_err("SDL_init");

    //WINDOW CREATION
    int w = 640;
    int h = 480;
    SDL_Window* SDLwin = SDL_CreateWindow("win1", w, h ,  0);
    if(SDLwin == NULL)
    check_err("SDL_CreateWindow");

    /////// SDL presentation = software
    /////////// actual renderer later = CUDA GPU
    // //RENDERER
    // SDL_Renderer* SDLrend = SDL_CreateRenderer(SDLwin,"software");
    // if(SDLrend == NULL)
    // check_err("SDL_CreateRenderer");

    // //TEXTURE
    // SDL_Texture* SDLtext =  SDL_CreateTexture(SDLrend,  SDL_PIXELFORMAT_RGBA32 , SDL_TEXTUREACCESS_STREAMING , w, h);
    // if(SDLtext == NULL)
    // check_err("SDL_CreateTexture");

    //So alternatively we use SDL_GetWindowSurface() provides a direct software context 
    // It is managed by SDL itself so no manual freeing
    // #define SDL_HINT_FRAMEBUFFER_ACCELERATION "0"
    SDL_Surface *SDLsurf = SDL_GetWindowSurface(SDLwin);
    if(SDLsurf == NULL)
    check_err("SDL_GetWindowSurface");
    
    //SDL EVENT
    SDL_Event event;
    bool running = true;
    cuda_state cs = cuda_init(w,h);
    int blockx = 16;
    int blocky = 16;
    float theta = 0;
    float omega = 0.5f; //useing 1 rad/s
    auto last_time = std::chrono::steady_clock::now();
    
    while(running)
    {while(SDL_PollEvent(&event))
        {
        if(event.type == SDL_EVENT_QUIT)
            running = false;
        }
        auto curr_time = std::chrono::steady_clock::now();
       	float dt = std::chrono::duration<float>(curr_time - last_time).count();
        last_time = curr_time;
        // dtheta/dt = omega
        theta += omega * dt;
        auto[out1,out2] = cuda_render(cs,blockx,blocky,(u32*)SDLsurf->pixels,(size_t)SDLsurf->pitch,theta);
        std::cout << out1 << "\n" << out2 << "\n";
        // void *pixels; int pitch;
        // bool SDLtextlock = SDL_LockTexture(SDLtext,NULL,&pixels,&pitch);
        // if(!SDLtextlock)
            // check_err("SDL_TextureLock");
        //
        // for(int y = 0; y < h ; y++)
        // {
        //     int row = SDLsurf->pitch * y ;
        //     for(int x =0 ; x < w ; x++)
        //     {
        //         uint32_t *rowptr =(uint32_t *)((unsigned char *)SDLsurf->pixels+ y * SDLsurf->pitch);
        //         rowptr[x] = 0x00FF0000;
        //     }
        // }
        auto a  = std::chrono::steady_clock::now();
        SDL_UpdateWindowSurface(SDLwin);
        auto b  = std::chrono::steady_clock::now();
        float present_ms = std::chrono::duration<float, std::milli>(b - a).count();
        std::cout << present_ms<< "\n" ;

        // SDL_UnlockTexture(SDLtext);
           // SDL_RenderTexture(SDLrend, SDLtext, NULL, NULL);
        // SDL_RenderPresent(SDLrend);
    }
    cuda_destroy(cs);
    // SDL_DestroyTexture(SDLtext);
    // SDL_DestroyRenderer(SDLrend);
    SDL_DestroyWindow(SDLwin);
    SDL_Quit();
    return 0;
}
