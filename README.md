# CUDA 3D Software Renderer

A small CUDA-based wireframe renderer that rotates and projects a 3D cube without using OpenGL, Vulkan, or Direct3D.

CUDA performs the framebuffer generation, while SDL3 provides a software window surface for presentation. This began as an experiment to understand how 3D coordinates eventually become pixels rather than relying on an existing graphics API to perform the entire pipeline.

## Current milestone

The current renderer:

- represents a cube using eight 3D vertices;
- rotates the vertices around the X-axis;
- applies perspective projection;
- discovers the cube's 12 edges from the vertex coordinates;
- rasterizes the projected edges using the Digital Differential Analyzer (DDA) algorithm;
- gives each CUDA thread ownership of one framebuffer pixel;
- copies the completed framebuffer from device memory into an SDL3 window surface;
- measures kernel, device-to-host copy, and SDL presentation time separately.

The result is a continuously rotating wireframe cube rendered into a 640 x 480 framebuffer.

## Rendering pipeline

```text
3D cube vertices
        |
        v
X-axis rotation
        |
        v
Perspective projection
        |
        v
DDA edge sampling
        |
        v
CUDA device framebuffer
        |
        | cudaMemcpy2D
        v
SDL3 software surface
        |
        v
Window presentation
```

SDL is not being used as the 3D renderer. It only owns the host-side window surface and presents the pixels produced by CUDA.

## Projection

A translated 3D point is projected using:

```text
projected_x = x / (z + z_translation)
projected_y = y / (z + z_translation)
```

The projected coordinates are then scaled and converted from a conventional centre-origin coordinate system into framebuffer coordinates:

```text
screen_x = width  / 2 + projected_x * scale
screen_y = height / 2 - projected_y * scale
```

The subtraction in `screen_y` accounts for screen coordinates increasing downward.

## CUDA execution model

The framebuffer uses a two-dimensional CUDA launch. With the default 16 x 16 block size:

```cpp
dim3 block(16, 16);
dim3 grid(
    (width  + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
```

Each thread corresponds to one output pixel:

```cpp
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;
pixel = y * width + x;
```

A thread writes only its own pixel, which avoids multiple threads racing to update the same framebuffer location.

## Current limitation

The simple ownership model comes with a major performance limitation: every pixel thread repeats the cube's geometry work. Each thread reconstructs and rotates the vertices, finds the cube edges, walks their DDA samples, and finally checks whether one of those samples belongs to its pixel.

This makes the first version easy to reason about and race-free, but it performs a large amount of redundant computation. The project therefore does **not** currently claim a performance improvement merely because it uses CUDA.

A future version will separate the pipeline into more appropriate stages, for example:

1. clear the framebuffer;
2. transform the eight vertices once;
3. project them once;
4. rasterize the 12 edges in parallel;
5. later move toward triangle rasterization, clipping, and depth testing.

## Timing

Three parts of a frame are measured independently:

- CUDA event timing for the rendering kernel;
- CPU wall-clock timing around `cudaMemcpy2D()`;
- CPU wall-clock timing around `SDL_UpdateWindowSurface()`.

The current program prints raw per-frame measurements. Averaged statistics and less intrusive reporting will be added later because printing every frame can itself disturb timing.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit and `nvcc`
- SDL3
- `pkg-config`
- A C++17-capable host compiler

The project was developed on an RTX 2070 Super, so the default Makefile architecture is `sm_75`.

## Build

```bash
make
```

To select a different CUDA architecture:

```bash
make CUDA_ARCH=sm_86
```

Run the renderer:

```bash
make run
```

Remove generated build output:

```bash
make clean
```

The equivalent direct build command is:

```bash
nvcc -std=c++17 -O2 -arch=sm_75 \
    cube.cu sdl.cpp \
    -o cube \
    $(pkg-config --cflags --libs sdl3)
```

## Project structure

```text
cube.cu         CUDA framebuffer, transformation, projection and DDA logic
sdl.cpp         SDL3 window, frame loop, timing and presentation
gpu_renderer.h  Shared renderer types and host-facing CUDA functions
Makefile        Configurable build, run and clean targets
```

## Planned work

- remove redundant per-pixel geometry work;
- separate transformation and rasterization stages;
- support rotation around multiple axes;
- add triangle filling;
- add a depth buffer;
- add clipping for geometry behind or too close to the camera;
- compare alternative CUDA work mappings with measurements;
- add a recorded GIF or video of the renderer.

## Scope

This is currently a CUDA wireframe software renderer, not a complete 3D rasterization pipeline. It does not yet implement filled triangles, depth testing, clipping, shading, or a general model/view/projection matrix system.

This README was written with AI assistance based on the project's source code and the author's implementation notes.
