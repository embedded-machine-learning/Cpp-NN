





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
| Name S-Edge- | downsampling factors     | Size [Byte] Ssm  | Size [Byte] Decoder |  Perm. Memory [Byte]  | Temp. Memory [Byte] | SUM [MFLOP]  | MCU Time [µs]  | Speed [MFLOP/s]  | Acc. [%] (Cpp on PC) |
|:-------------|:-------------------------|-----------------:|--------------------:|----------------------:|--------------------:|-------------:|---------------:|-----------------:|---------------------:|
| Full RZOH    | [1, 1, 1, 1, 1, 1]       |           549660 |               13588 |                  3456 |                 768 |      4406.95 |      103532350 |            42.57 |                95.19 |
| Full RZOH    | [1, 1, 1, 2, 4, 12]      |           549660 |               13588 |                  3456 |                 768 |      2309.48 |       56121220 |            41.15 |                95.22 |
| Full RZOH    | [1, 1, 4, 8, 16, 32]     |           549660 |               13588 |                  3456 |                 768 |      1198.31 |       29193430 |            41.05 |                94.99 |
| Full RZOH    | [1, 2, 14, 14, 42, 42]   |           549660 |               13588 |                  3456 |                 768 |       573.51 |       13942369 |            41.13 |                94.08 |
| L RZOH       | [1, 1, 1, 1, 1, 1]       |           215196 |                9108 |                  2944 |                 480 |      1721.41 |       35428430 |            48.59 |                94.11 |
| L RZOH       | [1, 1, 1, 2, 2, 38]      |           215196 |                9108 |                  2944 |                 480 |       964.38 |       20032550 |            48.14 |                94.19 |
| L RZOH       | [1, 2, 2, 8, 8, 48]      |           215196 |                9108 |                  2944 |                 480 |       413.89 |        8639978 |            47.90 |                93.99 |
| L RZOH       | [1, 3, 3, 6, 30, 60]     |           215196 |                9108 |                  2944 |                 480 |       295.64 |        6214082 |            47.58 |                93.09 |
| S no RZOH    | [1, 1, 1, 1, 1, 1]       |            71564 |                9108 |                  1104 |                 472 |       572.44 |       12384350 |            46.22 |                93.37 |
| S no RZOH    | [1, 1, 1, 1, 2, 22]      |            71564 |                9108 |                  1104 |                 472 |       237.21 |        5608158 |            42.30 |                93.47 |
| S no RZOH    | [1, 1, 3, 6, 6, 42]      |            71564 |                9108 |                  1104 |                 472 |        93.71 |        2303419 |            40.68 |                93.02 |
| S RZOH       | [1, 1, 1, 1, 1, 1]       |            71564 |                9108 |                  1104 |                 472 |       572.50 |       12384349 |            46.23 |                93.08 |
| S RZOH       | [1, 1, 1, 1, 9, 63]      |            71564 |                9108 |                  1104 |                 472 |       183.72 |        4427505 |            41.49 |                93.18 |
| S RZOH       | [1, 2, 2, 6, 18, 54]     |            71564 |                9108 |                  1104 |                 472 |        71.21 |        1722619 |            41.34 |                93.05 |
| S RZOH       | [1, 2, 4, 16, 32, 64]    |            71564 |                9108 |                  1104 |                 472 |        47.29 |        1161156 |            40.73 |                92.02 |
| S RZOH       | [1, 4, 4, 16, 32, 64]    |            71564 |                9108 |                  1104 |                 472 |        39.25 |         935281 |            41.96 |                91.06 |
| S RZOH       | [1, 4, 4, 16, 64, 64]    |            71564 |                9108 |                  1104 |                 472 |        37.44 |         894496 |            41.86 |                90.19 |
| Tiny no RZOH | [1, 1, 1]                |            24144 |                9108 |                   704 |                 384 |       193.61 |        4398077 |            44.02 |                90.50 |
| Tiny no RZOH | [1, 1, 5]                |            24144 |                9108 |                   704 |                 384 |        65.29 |        1595464 |            40.92 |                90.62 |
| Tiny no RZOH | [1, 2, 22]               |            24144 |                9108 |                   704 |                 384 |        26.98 |         667912 |            40.39 |                90.18 |
| Tiny no RZOH | [1, 4, 28]               |            24144 |                9108 |                   704 |                 384 |        18.65 |         450114 |            41.43 |                88.96 |
| Tiny RZOH    | [1, 1, 1]                |            24144 |                9108 |                   704 |                 384 |       193.59 |        4398077 |            44.02 |                89.41 |
| Tiny RZOH    | [1, 2, 6]                |            24144 |                9108 |                   704 |                 384 |        46.42 |        1109800 |            41.82 |                89.51 |
| Tiny RZOH    | [1, 5, 20]               |            24144 |                9108 |                   704 |                 384 |        19.58 |         468652 |            41.78 |                89.12 |
| Tiny RZOH    | [1, 9, 54]               |            24144 |                9108 |                   704 |                 384 |        12.13 |         289062 |            41.98 |                88.01 |
| Tiny RZOH    | [1, 11, 55]              |            24144 |                9108 |                   704 |                 384 |        11.53 |         271840 |            42.40 |                87.11 |
| Tiny RZOH    | [2, 8, 64]               |            24144 |                9108 |                   704 |                 384 |         8.97 |         219953 |            40.76 |                80.35 |
| Tiny RZOH    | [3, 12, 48]              |            24144 |                9108 |                   704 |                 384 |         7.66 |         186324 |            41.10 |                70.32 |
