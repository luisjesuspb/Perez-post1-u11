# Post-Contenido 1 â€” Benchmark CUDA (CPU vs. GPU) y AnÃ¡lisis Comparativo de Aceleradores de IA

**Asignatura:** Arquitectura de Computadores â€” Unidad 11  
**Estudiante:** Luis JesÃºs PÃ©rez (`luisjesuspb`)  
**Repositorio:** `Perez-post1-u11`  
**Documento Parte 2:** [`docs/analisis-comparativo.md`](docs/analisis-comparativo.md)

---

## 1. DescripciÃ³n del Entorno de EjecuciÃ³n

Las pruebas empÃ­ricas de la Parte 1 se ejecutaron sobre una instancia acelerada por hardware NVIDIA en Google Colab:

| Componente | EspecificaciÃ³n |
|:-----------|:---------------|
| **GPU Aceleradora** | NVIDIA Tesla T4 (Arquitectura Turing, 2,560 CUDA Cores, 320 Tensor Cores, 15,360 MiB GDDR6) |
| **Driver NVIDIA / CUDA** | Driver `580.82.07` Â· CUDA Toolkit `13.0` (`nvcc V13.0.88`) |
| **Sistema Operativo** | Ubuntu Linux (entorno x86_64) |
| **Compilador y Flags** | `nvcc -O2 -Xcompiler -Wall` |

![VerificaciÃ³n de entorno GPU Tesla T4 y CUDA](capturas/01_entorno_cuda.png)

---

## 2. Comandos de CompilaciÃ³n y EjecuciÃ³n

```bash
# 1. Suma de vectores (CPU vs GPU)
nvcc -O2 -Xcompiler -Wall -o vectorAdd src/vectorAdd.cu && ./vectorAdd

# 2. MultiplicaciÃ³n de matrices (Naive vs Shared Memory Tiling)
nvcc -O2 -Xcompiler -Wall -o matMul src/matMul.cu && ./matMul

# 3. Suma de vectores con transferencia asÃ­ncrona y CUDA Streams
nvcc -O2 -Xcompiler -Wall -o vectorAddStream src/vectorAddStream.cu && ./vectorAddStream
```

---

## 3. Resultados Experimentales

### 3.1. Suma de Vectores (`src/vectorAdd.cu`)

| Elementos ($N$) | TamaÃ±o por Vector | CPU (`ms`) | GPU Kernel (`ms`) | GPU Total c/ `cudaMemcpy` (`ms`) | Speedup Kernel (CPU/GPU) | Errores |
|----------------:|------------------:|-----------:|------------------:|---------------------------------:|-------------------------:|--------:|
| $1,048,576$ ($1\text{M}$) | 4.0 MB | 3.11 ms | 0.06 ms | 6.03 ms | 51.83x | 0 |
| $4,194,304$ ($4\text{M}$) | 16.0 MB | 14.72 ms | 0.19 ms | 21.17 ms | 77.47x | 0 |
| $16,777,216$ ($16\text{M}$) | 64.0 MB | 56.84 ms | 0.77 ms | 81.28 ms | 73.82x | 0 |

![Benchmark vectorAdd](capturas/02_vectorAdd.png)

### 3.2. MultiplicaciÃ³n de Matrices con Tiling (`src/matMul.cu`, `TILE = 16`)

| DimensiÃ³n ($N \times N$) | CPU (`ms`) | GPU NaÃ¯ve (`ms`) | GPU Tiled Shared Memory (`ms`) | Speedup Tiling vs. NaÃ¯ve | Speedup Tiled vs. CPU | Error MÃ¡ximo |
|----------------:|-----------:|---------------:|-------------------------------:|-------------------------:|---------------------:|-------------:|
| $512 \times 512$ | 96.53 ms | 0.65 ms | 0.42 ms | **1.56x** | 229.83x | `4.58e-05` |
| $1024 \times 1024$ | 641.41 ms | 4.93 ms | 3.09 ms | **1.60x** | 207.58x | `1.07e-04` |

![Benchmark matMul Naive vs Tiling](capturas/03_matMul.png)

### 3.3. Transferencia AsÃ­ncrona con CUDA Streams (`src/vectorAddStream.cu`, $N = 16\text{M}$)

| Modalidad de EjecuciÃ³n | Tiempo Total (`ms`) | ReducciÃ³n de Tiempo | Errores |
|:-----------------------|--------------------:|--------------------:|--------:|
| Baseline SÃ­ncrono (`cudaMemcpy` bloqueante + kernel) | 16.85 ms | â€” | 0 |
| AsÃ­ncrono con `NSTREAMS = 4` (memoria *pinned* + `cudaMemcpyAsync`) | 13.04 ms | **22.6 %** | 0 |

![Benchmark vectorAddStream](capturas/04_vectorAddStream.png)

---

## 4. AnÃ¡lisis de Resultados

Los resultados empÃ­ricos demuestran cÃ³mo la arquitectura SIMT de la GPU Tesla T4 acelera drÃ¡sticamente cargas masivamente paralelas conforme crece el tamaÃ±o del problema. En `vectorAdd`, el tiempo aislado de ejecuciÃ³n del kernel supera a la CPU hasta en **77.47x** ($0.19\text{ ms}$ frente a $14.72\text{ ms}$ para $4\text{M}$ elementos), debido a que miles de hilos CUDA ejecutan sumas independientes en paralelo ocultando la latencia de memoria global GDDR6. Sin embargo, al incluir el tiempo de transferencia de datos entre host y dispositivo (`cudaMemcpy` sobre el bus PCIe), el tiempo total de GPU ($6.03\text{ ms}$ para $1\text{M}$ y $81.28\text{ ms}$ para $16\text{M}$ usando memoria paginable con `malloc`) supera al tiempo de ejecuciÃ³n en CPU. Esto ocurre porque `vectorAdd` realiza apenas 1 operaciÃ³n de punto flotante por cada 12 bytes transferidos, haciendo que la latencia y el ancho de banda del bus PCIe dominen el tiempo total cuando el cÃ³mputo es trivial.

Por el contrario, en la multiplicaciÃ³n de matrices (`matMul`), la complejidad computacional es $\mathcal{O}(N^3)$ frente a $\mathcal{O}(N^2)$ datos transferidos, lo que amortiza completamente el costo del bus PCIe y permite que la GPU supere a la CPU en mÃ¡s de **207x** para $N=1024$. AdemÃ¡s, incorporar *tiling* con memoria compartida (`__shared__` de $16 \times 16$) reduce los accesos redundantes a memoria global en un factor de 16, logrando una mejora adicional consistente de **1.56x a 1.60x** sobre el kernel *naÃ¯ve* y manteniendo el error numÃ©rico por debajo de $1.07 \times 10^{-4}$.

---

## 5. Decisiones TÃ©cnicas (Parte 1)

### DecisiÃ³n TÃ©cnica 1: Â¿Conviene `cudaMemcpyAsync` con streams o transferencia sÃ­ncrona para `vectorAdd`?

Para $N = 16\text{M}$ elementos en precisiÃ³n simple (`float`), el volumen total de datos movidos a travÃ©s del bus PCIe entre host y dispositivo asciende a **192 MB** (64 MB para `h_A`, 64 MB para `h_B` y 64 MB de retorno para `h_C`). Sobre un enlace PCIe Gen3/Gen4 x16 con un ancho de banda efectivo real cercano a 10â€“12 GB/s tras descontar la sobrecarga de protocolo, transferir esos 192 MB requiere aproximadamente **16 a 19 ms** (coincidente con los `16.85 ms` medidos en el baseline sÃ­ncrono con memoria *pinned*). Al contrastar esa cifra con el tiempo aislado de cÃ³mputo del kernel (`0.77 ms` en Tesla T4), resulta evidente que `vectorAdd` es una carga estrictamente **memory-bound**, con una intensidad aritmÃ©tica de apenas $1/12 \approx 0.083\text{ FLOP/byte}$.

Dado que el uso de *streams* no reduce la cantidad total de bytes que atraviesan el bus PCIe, sino que Ãºnicamente permite solapar las transferencias DMA con la ejecuciÃ³n del kernel, el ahorro mÃ¡ximo teÃ³rico estÃ¡ acotado por la duraciÃ³n de la fase mÃ¡s corta: los `0.77 ms` de cÃ³mputo (mÃ¡s el solapamiento parcial entre sub-bloques H2D/D2H). Aunque en la prueba experimental con memoria *pinned* se obtuvo una reducciÃ³n del **22.6 %** (`16.85 ms` a `13.04 ms`), gran parte de esa ganancia proviene de habilitar transferencias DMA directas sin copias intermedias de staging y de solapar los extremos de los *chunks*. En consecuencia, para una operaciÃ³n puramente *memory-bound* aislada como `vectorAdd`, el beneficio de implementar *streams* es limitado; su adopciÃ³n se vuelve verdaderamente determinante en operaciones con alta intensidad aritmÃ©tica como `matMulTiled`, donde el tiempo de cÃ³mputo $\mathcal{O}(N^3)$ de cada submatriz es comparable o superior al tiempo de transferencia PCIe, permitiendo ocultar casi por completo la latencia de comunicaciÃ³n.

### DecisiÃ³n TÃ©cnica 2: Â¿CuÃ¡ntos streams usar y quÃ© limita la ganancia al aumentar su nÃºmero?

Incrementar `NSTREAMS` de 4 a 16 manteniendo fijo $N = 16\text{M}$ **no** produce una reducciÃ³n proporcional del tiempo total de ejecuciÃ³n. El lÃ­mite fundamental reside en los recursos fÃ­sicos compartidos del hardware: el ancho de banda agregado del bus PCIe y el nÃºmero de motores de copia DMA (*copy engines*) disponibles en la GPU. La mayorÃ­a de las tarjetas NVIDIA de grado centro de datos y consumo (incluida la Tesla T4) exponen Ãºnicamente **1 o 2 copy engines** fÃ­sicos independientes del motor de cÃ³mputo (permitiendo como mÃ¡ximo una transferencia *Host-to-Device* y otra *Device-to-Host* simultÃ¡neas si el bus bidireccional lo soporta).

Por lo tanto, basta con un nÃºmero reducido de *streams* â€”tÃ­picamente entre **2 y 4**â€” para saturar el *pipeline* de transferencia y mantener ocupados tanto los motores DMA como los multiprocesadores de *streaming* (SM). AÃ±adir 16 o mÃ¡s *streams* no crea canales fÃ­sicos adicionales en el bus PCIe ni multiplica el ancho de banda disponible; el volumen total de 192 MB sigue limitado por el mismo techo fÃ­sico de transferencia. En cambio, fragmentar excesivamente los arreglos en bloques muy pequeÃ±os introduce rendimientos decrecientes debido al *overhead* de planificaciÃ³n de la cola de comandos en el driver CUDA y a una menor eficiencia de las transacciones DMA por rÃ¡faga.

---


