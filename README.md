# C++20 Branch
The original C++17 has been deprecated, as requires and concepts are just too important for readability and the ability to conditionally enable code.

# CPU
## Compiler 
It was tested with Clang++-18 [Ubuntu clang version 18.1.6 (++20240518023229+1118c2e05e67-1~exp1~20240518143321.130)] .
As well as g++-12 [gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04)].

## Flags
### LLVM
#### OPT
-Os 
-std=c++20 
-march=native 
-fconstexpr-steps=100000000 

[//]: # (increaseing constexpr limit might not be nessesary anymore)

#### Warnings/Info
-Wall 
-Wpedantic 
-Winline 
-fstack-usage 

#### pybind
-shared 
-fPIC $(python3 -m pybind11 --includes) 

### GCC
The new version is not compadiable with -Os anymore as for some reason functions are not inlined anymore correctly. Furthermore, its not compadable with the AXV2 MAC operations implemented here.

#### OPT
-O3
-std=c++20 
-march=native 
#### Warnings/Info
-Wall 
-Wpedantic 
-fstack-usage 
-Wno-inline 
-fdump-tree-optimized
 
#### pybind
-shared 
-fPIC $(python3 -m pybind11 --includes)




# ARM
## Compiler
We used STM32CubeIDE 1.15.0 with the Tool chain: 'GNU Tools for STM32 (12.3.rel1)'.

We're using C++20
## Flags
We're compiling with O3 or Ofast. Os does not always inline ´__attribute__((always_inline))´ functions, which breaks design choises of this framework. Most data access is behind 3-4 of such functions, and only if they are inlined they can be fully resolved at compiletime leading to zero overhead and essentailly raw pointer operations. 

### GCC Compiler 
-flto
-ffast-math 
-funsafe-math-optimizations 
-fno-math-errno 
-fno-trapping-math
-frename-registers 
-fprefetch-loop-arrays 
-fpredictive-commoning 
-fgcse-after-reload
-falign-loops=4
-ffunction-sections
-fdata-sections

### G++ Compiler
-flto
-Winline
-fstack-usage
-ffast-math 
-funsafe-math-optimizations 
-fno-math-errno 
-fno-trapping-math
-fno-exceptions 
-fno-rtti
-frename-registers 
-fprefetch-loop-arrays 
-fpredictive-commoning 
-fgcse-after-reload
-falign-loops=4
-ftree-vectorize 
-fvect-cost-model=cheap 
-fsched-pressure
-ffunction-sections
-fdata-sections
-fno-threadsafe-statics
-fno-cxa-atexit
-nostdlib++ 

[//]: # (-nostdlib++  Doesnt Do anything in CubeIDE as -lstdc++ is automatically added afterwards)


### Linker
-flto
-nostdlib++ 

[//]: # (-nostdlib++  Doesnt Do anything in CubeIDE as -lstdc++ is automatically added afterwards)

## Perfrormance
### SEdge
Tiny has slightly inreased in temp. memory compared to the C++17 variation, as it now also has a full copy of the outputs (float[35]) present, even though it does not use it.

| Name S-Edge- | downsampling factors     |Network Size<br>[Byte]|Perm. Memory<br>[Byte]|Temp. Memory<br>[Byte]|Total FLOPS<br>[MFLOP]|MCU Time<br>[s]|Speed<br>[MFLOP/s]|Acc. (Cpp on PC)<br>[%]|
|:-------------|:-------------------------|--------------------:|----------------------:|---------------------:|--------------------:|---------------:|-----------------:|----------------------:|
| Full RZOH    | [1, 1, 1, 1, 1, 1]       |              563528 |                  3456 |                  768 |             4406.95 |       *87.0000*|           *50.65*|                 95.19 |
| Full RZOH    | [1, 1, 1, 2, 4, 12]      |              563528 |                  3456 |                  768 |             2309.48 |       *47.0000*|           *49.14*|                 95.22 |
| Full RZOH    | [1, 1, 4, 8, 16, 32]     |              563528 |                  3456 |                  768 |             1198.31 |        23.5652 |            50.85 |                 94.99 |
| Full RZOH    | [1, 2, 14, 14, 42, 42]   |              563528 |                  3456 |                  768 |              573.51 |      **11.0938**|        **51.70**|                94.08 |
| L RZOH       | [1, 1, 1, 1, 1, 1]       |              224592 |                  2944 |                  480 |             1721.41 |        35.1360 |            48.99 |                 94.11 |
| L RZOH       | [1, 1, 1, 2, 2, 38]      |              224592 |                  2944 |                  480 |              964.38 |        19.4856 |            49.49 |                 94.19 |
| L RZOH       | [1, 2, 2, 8, 8, 48]      |              224592 |                  2944 |                  480 |              413.89 |         8.4339 |            49.07 |                 93.99 |
| L RZOH       | [1, 3, 3, 6, 30, 60]     |              224592 |                  2944 |                  480 |              295.64 |       **6.0680**|         **48.72**|                93.09 |
| S no RZOH    | [1, 1, 1, 1, 1, 1]       |               80984 |                  1104 |                  472 |              572.50 |        11.6736 |            49.04 |                 93.37 |
| S no RZOH    | [1, 1, 1, 1, 2, 22]      |               80984 |                  1104 |                  472 |              237.21 |         5.0897 |            46.61 |                 93.47 |
| S no RZOH    | [1, 1, 3, 6, 6, 42]      |               80984 |                  1104 |                  472 |               93.71 |         2.0427 |            45.88 |                 93.02 |
| S RZOH       | [1, 1, 1, 1, 1, 1]       |               80984 |                  1104 |                  472 |              572.50 |        11.6736 |            49.04 |                 93.08 |
| S RZOH       | [1, 1, 1, 1, 9, 63]      |               80984 |                  1104 |                  472 |              183.72 |         4.0569 |            45.28 |                 93.18 |
| S RZOH       | [1, 2, 2, 6, 18, 54]     |               80984 |                  1104 |                  472 |               71.21 |         1.5736 |            45.25 |                 93.05 |
| S RZOH       | [1, 2, 4, 16, 32, 64]    |               80984 |                  1104 |                  472 |               47.29 |         1.1149 |            42.42 |                 92.02 |
| S RZOH       | [1, 4, 4, 16, 32, 64]    |               80984 |                  1104 |                  472 |               39.25 |       **0.8689**|         **45.17**|                91.06 |
| S RZOH       | [1, 4, 4, 16, 64, 64]    |               80984 |                  1104 |                  472 |               37.44 |         0.8374 |            44.71 |                 90.19 |
| Tiny no RZOH | [1, 1, 1]                |               33424 |                   704 |                  396 |              193.61 |         4.1833 |            46.28 |                 90.50 |
| Tiny no RZOH | [1, 1, 5]                |               33424 |                   704 |                  396 |               65.29 |         1.3880 |            47.04 |                 90.62 |
| Tiny no RZOH | [1, 2, 22]               |               33424 |                   704 |                  396 |               26.98 |         0.6175 |            43.69 |                 90.18 |
| Tiny no RZOH | [1, 4, 28]               |               33424 |                   704 |                  396 |               18.65 |         0.4111 |            45.36 |                 88.96 |
| Tiny RZOH    | [1, 1, 1]                |               33424 |                   704 |                  396 |              193.61 |         4.1833 |            46.28 |                 89.41 |
| Tiny RZOH    | [1, 2, 6]                |               33424 |                   704 |                  396 |               46.42 |         1.0290 |            45.11 |                 89.51 |
| Tiny RZOH    | [1, 5, 20]               |               33424 |                   704 |                  396 |               19.58 |         0.4307 |            45.46 |                 89.12 |
| Tiny RZOH    | [1, 9, 54]               |               33424 |                   704 |                  396 |               12.13 |         0.2660 |            45.62 |                 88.01 |
| Tiny RZOH    | [1, 11, 55]              |               33424 |                   704 |                  396 |               11.53 |       **0.2518**|         **45.78**|                87.11 |
| Tiny RZOH    | [2, 8, 64]               |               33424 |                   704 |                  396 |                8.97 |         0.2092 |            42.85 |                 80.35 |
| Tiny RZOH    | [3, 12, 48]              |               33424 |                   704 |                  396 |                7.66 |         0.1816 |            42.17 |                 70.32 |

*cursive* means they where measured with a smartphone stop watch, and are therefore highly inaccurate

**bold** are optimization points, the configurations that where hand tuned for maximum performance, which are then applied for all other configurations of that network.


