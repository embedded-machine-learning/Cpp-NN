
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
-finline-limit=1000000000 
--param max-unrolled-insns=1000000000 
--param max-inline-insns-single=1000000000 
--param inline-unit-growth=1000000 
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
It does not compile with 13.3.rel1 .

We're using C++20
## Flags
We're compiling with O3 or Ofast. Os breaks the compiler sometimes as it stops inline some parts, which leads to insane overhead. Othertimes its realy slow as it didn't inline data loading (and branches instead). 

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
| Full RZOH    | [1, 1, 1, 1, 1, 1]       |              563528 |                  3456 |                  768 |             4406.95 |       *93.5000 |           *47.13 |                 95.19 |
| Full RZOH    | [1, 1, 1, 2, 4, 12]      |              563528 |                  3456 |                  768 |             2309.48 |       *51.0000 |           *45.28 |                 95.22 |
| Full RZOH    | [1, 1, 4, 8, 16, 32]     |              563528 |                  3456 |                  768 |             1198.31 |        26.1651 |            45.80 |                 94.99 |
| Full RZOH    | [1, 2, 14, 14, 42, 42]   |              563528 |                  3456 |                  768 |              573.51 |        12.2525 |            46.81 |                 94.08 |
| L RZOH       | [1, 1, 1, 1, 1, 1]       |              224592 |                  2944 |                  480 |             1721.41 |       *36.5000 |           *47.16 |                 94.11 |
| L RZOH       | [1, 1, 1, 2, 2, 38]      |              224592 |                  2944 |                  480 |              964.38 |        20.1751 |            47.80 |                 94.19 |
| L RZOH       | [1, 2, 2, 8, 8, 48]      |              224592 |                  2944 |                  480 |              413.89 |         8.6694 |            47.74 |                 93.99 |
| L RZOH       | [1, 3, 3, 6, 30, 60]     |              224592 |                  2944 |                  480 |              295.64 |         6.1743 |            47.88 |                 93.09 |
| S no RZOH    | [1, 1, 1, 1, 1, 1]       |               80984 |                  1104 |                  472 |              572.50 |        13.7922 |            41.51 |                 93.37 |
| S no RZOH    | [1, 1, 1, 1, 2, 22]      |               80984 |                  1104 |                  472 |              237.21 |         5.5013 |            43.12 |                 93.47 |
| S no RZOH    | [1, 1, 3, 6, 6, 42]      |               80984 |                  1104 |                  472 |               93.71 |         2.1869 |            42.85 |                 93.02 |
| S RZOH       | [1, 1, 1, 1, 1, 1]       |               80984 |                  1104 |                  472 |              572.50 |        13.7922 |            41.51 |                 93.08 |
| S RZOH       | [1, 1, 1, 1, 9, 63]      |               80984 |                  1104 |                  472 |              183.72 |         4.2273 |            43.46 |                 93.18 |
| S RZOH       | [1, 2, 2, 6, 18, 54]     |               80984 |                  1104 |                  472 |               71.21 |         1.6750 |            42.51 |                 93.05 |
| S RZOH       | [1, 2, 4, 16, 32, 64]    |               80984 |                  1104 |                  472 |               47.29 |         1.1053 |            42.79 |                 92.02 |
| S RZOH       | [1, 4, 4, 16, 32, 64]    |               80984 |                  1104 |                  472 |               39.25 |         0.9313 |            42.14 |                 91.06 |
| S RZOH       | [1, 4, 4, 16, 64, 64]    |               80984 |                  1104 |                  472 |               37.44 |         0.8669 |            43.19 |                 90.19 |
| Tiny no RZOH | [1, 1, 1]                |               33424 |                   704 |                  396 |              193.61 |         4.7696 |            40.59 |                 90.50 |
| Tiny no RZOH | [1, 1, 5]                |               33424 |                   704 |                  396 |               65.29 |         1.5509 |            42.10 |                 90.62 |
| Tiny no RZOH | [1, 2, 22]               |               33424 |                   704 |                  396 |               26.98 |         0.6267 |            43.05 |                 90.18 |
| Tiny no RZOH | [1, 4, 28]               |               33424 |                   704 |                  396 |               18.65 |         0.4413 |            42.26 |                 88.96 |
| Tiny RZOH    | [1, 1, 1]                |               33424 |                   704 |                  396 |              193.61 |         4.7696 |            40.59 |                 89.41 |
| Tiny RZOH    | [1, 2, 6]                |               33424 |                   704 |                  396 |               46.42 |         1.1131 |            41.70 |                 89.51 |
| Tiny RZOH    | [1, 5, 20]               |               33424 |                   704 |                  396 |               19.58 |         0.4705 |            41.62 |                 89.12 |
| Tiny RZOH    | [1, 9, 54]               |               33424 |                   704 |                  396 |               12.13 |         0.2841 |            42.71 |                 88.01 |
| Tiny RZOH    | [1, 11, 55]              |               33424 |                   704 |                  396 |               11.53 |         0.2829 |            40.75 |                 87.11 |
| Tiny RZOH    | [2, 8, 64]               |               33424 |                   704 |                  396 |                8.97 |         0.2148 |            41.75 |                 80.35 |
| Tiny RZOH    | [3, 12, 48]              |               33424 |                   704 |                  396 |                7.66 |         0.1899 |            40.33 |                 70.32 |

\* means they where measured with a smartphone stop watch, and are therefore highly inaccurate
