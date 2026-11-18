# Análisis Comparativo de Aceleradores de IA: GPU vs. TPU vs. NPU vs. FPGA

**Asignatura:** Arquitectura de Computadores — Unidad 11  
**Estudiante:** Luis Jesús Pérez (`luisjesuspb`)  
**Repositorio:** `Perez-post1-u11`

---

## 1. Introducción y Contexto Arquitectónico

El despliegue moderno de modelos de Inteligencia Artificial —desde redes convolucionales en el borde hasta Grandes Modelos de Lenguaje (LLM) en centros de datos— exige arquitecturas especializadas capaces de sostener tasas masivas de operaciones matriciales (`GEMM`, *General Matrix Multiply*) con alta eficiencia energética. Mientras que la CPU de propósito general optimiza la latencia por hilo mediante predicción de saltos y jerarquías profundas de caché, los aceleradores de IA sacrifican la lógica de control compleja para maximizar el paralelismo espacial y temporal sobre unidades aritméticas densas.

Este documento analiza comparativamente cuatro familias de aceleradores —**GPU**, **TPU**, **NPU** y **FPGA**— evaluando sus principios microarquitectónicos, jerarquías de memoria, modelos de programación y criterios de selección por escenario de carga de trabajo.

---

## 2. Tabla Comparativa de Arquitecturas

| Dimensión Arquitectónica | GPU (ej. NVIDIA H100 / Tesla T4) | TPU (ej. Google TPU v4 / v5e) | NPU (ej. Apple Neural Engine / Qualcomm Hexagon) | FPGA (ej. AMD Versal AI Core / Intel Agilex) |
|:-------------------------|:---------------------------------|:------------------------------|:-------------------------------------------------|:---------------------------------------------|
| **Paradigma de Cómputo** | **SIMT** (*Single Instruction, Multiple Threads*) + unidades matriciales dedicadas (Tensor Cores). | **Arreglo Sistólico** (*Systolic Array* 2D MXU, ej. $128 	imes 128$ MACs) con flujo rítmico sin accesos intermedios a registros. | **Motor de Flujo de Datos / Vectorial-Matricial** heterogéneo de ultra bajo consumo con MACs cuantizados. | **Lógica Reconfigurable Espacial** (LUTs, Flip-Flops, bloques DSP y bloques de memoria BRAM/URAM) con *pipelines* a medida. |
| **Jerarquía de Memoria** | Registros por hilo → *Shared Memory* / L1 configurable → Caché L2 compartida → **HBM3 / GDDR6** de alto ancho de banda. | Registros vectoriales → *Unified Buffer* (SRAM en chip de decenas de MB) → **HBM** (*High Bandwidth Memory*). | Buffers SRAM locales (*scratchpad* en chip) → **LPDDR5 / Memoria Unificada** compartida con el SoC. | Registros en Flip-Flops → **BRAM / UltraRAM** distribuida en el tejido lógico → DDR4 / HBM externa. |
| **Precisiones Soportadas** | FP64, FP32, TF32, FP16, BF16, FP8, INT8, INT4. | BF16, FP16, FP32 (acumulación), INT8, INT4 (optimizada para BF16/INT8). | INT8, INT4, INT2, FP16 (enfoque en inferencia cuantizada entera). | **Arbitraria / Personalizada** (INT1, INT2, INT3, INT4, punto fijo a medida, BF16, FP16). |
| **Consumo Típico (TDP)** | **70 W – 700 W** (70 W en T4; hasta 700 W en H100 SXM5). | **175 W – 450 W** por chip en pods de centro de datos. | **1 W – 15 W** (integrada en SoC móvil, portátil o automotriz). | **10 W – 225 W** (desde dispositivos embebidos hasta tarjetas PCIe de centro de datos). |
| **Ecosistema de Software** | **CUDA**, cuDNN, TensorRT, PyTorch, Triton (ecosistema más maduro y flexible). | **XLA** (*Accelerated Linear Algebra*), JAX, TensorFlow, PyTorch/XLA. | **Core ML**, SNPE / QNN, ONNX Runtime, LiteRT (TFLite), DirectML. | **Vitis AI**, OpenCL, HLS (*High-Level Synthesis*), Verilog/VHDL, FINN. |
| **Caso de Uso Óptimo** | Entrenamiento e inferencia general de LLM/visión, HPC científico y cargas dinámicas/irregulares. | Entrenamiento masivo distribuido e inferencia de alto *throughput* en modelos Transformer/CNN estáticos. | Inferencia local en tiempo real en dispositivos *edge* / móviles con restricción estricta de batería y privacidad. | Inferencia determinista de **ultra baja latencia** (*batch=1*), procesamiento de señales y precisión no estándar. |

---

## 3. Análisis Profundo por Arquitectura

### 3.1. GPU (*Graphics Processing Unit*)
Las GPUs modernas combinan miles de núcleos escalares organizables en *warps* (32 hilos en NVIDIA) con **Tensor Cores** capaces de ejecutar operaciones matriciales $D = A 	imes B + C$ en un solo ciclo de reloj por bloque. Su fortaleza radica en la **versatilidad programable**: mediante CUDA y memorias compartidas gestionadas por software (`__shared__`), permiten ejecutar tanto *kernels* irregulares (atención dispersa, grafos, simulaciones físicas) como multiplicaciones densas masivas respaldadas por memoria HBM3 de hasta $3.35	ext{ TB/s}$ (H100) e interconexiones NVLink de $900	ext{ GB/s}$.

### 3.2. TPU (*Tensor Processing Unit*)
Diseñada específicamente por Google para álgebra lineal de redes neuronales, la TPU sustituye la compleja planificación de hilos de la GPU por una o varias **Matrix Multiply Units (MXU)** basadas en **arreglos sistólicos**. En un arreglo sistólico de $128 	imes 128$, los pesos permanecen estacionarios en las celdas MAC mientras las activaciones fluyen en oleadas sincronizadas de celda a celda. Esto elimina decenas de miles de lecturas/escrituras hacia registros o cachés por cada operación, disparando la eficiencia energética (`TOPS/W`). Su contrapartida es la rigidez: requiere que el compilador **XLA** conozca las dimensiones de los tensores de antemano para planificar estáticamente el movimiento de datos entre *Unified Buffer* y HBM.

### 3.3. NPU (*Neural Processing Unit*)
Las NPUs son bloques IP integrados dentro de *Systems-on-Chip* (SoC) de teléfonos inteligentes, portátiles AI PC y sistemas embebidos. Están optimizadas para ejecutar redes neuronales previamente entrenadas y cuantizadas (INT8/INT4) consumiendo apenas unos pocos vatios. Para evitar el costo energético de acceder a la DRAM externa (que consume órdenes de magnitud más energía por bit que una operación MAC), las NPUs emplean memorias *scratchpad* SRAM internas y fusión de capas (*operator fusion*), manteniendo las activaciones intermedias dentro del silicio entre convoluciones o capas lineales consecutivas.

### 3.4. FPGA (*Field-Programmable Gate Array*)
A diferencia de las arquitecturas anteriores con instrucciones fijas, una FPGA permite sintetizar **circuitos hardware personalizados** directamente en silicio reconfigurable. En lugar de procesar lotes (*batches*) grandes para amortizar el lanzamiento de *kernels*, una FPGA construye un *pipeline* espacial profundo donde cada etapa de la red neuronal tiene su propio hardware dedicado y los datos fluyen continuamente sin esperar por un *batch*. Además, soporta tipos numéricos no estándar (como redes binarias INT1 o cuantización mixta INT3/INT5) con latencia determinista de microsegundos. Su principal desventaja es la complejidad de desarrollo en RTL/HLS y los largos tiempos de síntesis hardware.

---

## 4. Toma de Decisiones por Escenario (Parte 2)

### Escenario A: Entrenamiento de un LLM de 70B parámetros en centro de datos
* **Acelerador recomendado:** **Clúster de GPUs de centro de datos (NVIDIA H100/H200 con NVLink + InfiniBand)** o **Pod de TPUs (Google TPU v5p / v4)**.
* **Justificación técnica:** Un modelo de 70B parámetros requiere aproximadamente $140	ext{ GB}$ solo para almacenar los pesos en BF16/FP16, y más de $560	ext{ GB} - 1	ext{ TB}$ al incluir gradientes, estados del optimizador Adam (FP32) y activaciones. Esto exige paralelismo híbrido (tensorial, de *pipeline* y de datos) sobre memoria **HBM de alta capacidad** y redes de interconexión ultrarrápidas (`NVLink` a $900	ext{ GB/s}$ o topología Toroide 3D ICI en TPUs) para sincronizar gradientes (`All-Reduce`) sin cuellos de botella. Si el equipo requiere máxima flexibilidad de investigación (operadores personalizados en Triton, *Mixture-of-Experts* dinámicos), la **GPU** es superior; si el modelo tiene una arquitectura Transformer estándar a gran escala en Google Cloud, la **TPU** ofrece mayor eficiencia costo/FLOP gracias a sus arreglos sistólicos.

### Escenario B: Inferencia en tiempo real de visión por computador en un dron autónomo con batería limitada
* **Acelerador recomendado:** **NPU embebida de bajo consumo** (ej. Qualcomm Hexagon en plataforma RB5/RB6, NVIDIA Jetson Orin Nano en modo DLA/GPU de 7–15 W, o Intel Movidius/Core Ultra NPU).
* **Justificación técnica:** Un dron posee un presupuesto térmico y energético severamente restringido (típicamente $< 15	ext{ W}$ para el subsistema de cómputo) donde una GPU de servidor o TPU de centro de datos resulta inviable por peso y consumo. Una NPU operando con modelos cuantizados en **INT8** entrega decenas de TOPS consumiendo entre $2	ext{ W}$ y $10	ext{ W}$, garantizando inferencia local a más de $30	ext{–}60	ext{ FPS}$ sin depender de conectividad inalámbrica ni latencia de nube, maximizando la autonomía de vuelo.

### Escenario C: Detección de fraude financiero en transacciones con latencia máxima estricta de 2 ms (`batch = 1`)
* **Acelerador recomendado:** **FPGA de centro de datos** (ej. AMD Alveo / Versal o Intel Agilex con conectividad de red directa).
* **Justificación técnica:** En detección de fraude en línea o *high-frequency trading*, cada transacción llega de forma individual (`batch = 1`) y debe evaluarse en menos de $2	ext{ ms}$ con **latencia determinista** (sin picos de *tail latency*). Las GPUs sufren penalizaciones de latencia al transferir lotes unitarios por el bus PCIe y planificar *kernels* desde la CPU host, ya que están diseñadas para maximizar el *throughput* con *batches* grandes. Una FPGA puede conectarse directamente a la interfaz de red (`100GbE`), decodificar el paquete en hardware y atravesar el modelo de inferencia mediante un *pipeline* espacial continuo sin intervención del sistema operativo, garantizando tiempos de respuesta constantes del orden de decenas o cientos de microsegundos.

### Escenario D: Ejecución local de un asistente de voz y texto en un teléfono móvil sin conexión a internet
* **Acelerador recomendado:** **NPU integrada en el SoC móvil** (ej. Apple Neural Engine, Qualcomm Hexagon NPU o Google Tensor TPU móvil).
* **Justificación técnica:** El procesamiento 100 % local (*on-device*) exige privacidad total de los datos del usuario, cero latencia de red y mínimo impacto en la batería y temperatura del teléfono. Las NPUs móviles incorporan motores especializados para atención de Transformers cuantizados (`INT4` para pesos y `INT8` para activaciones) y acceso directo a la memoria unificada LPDDR5 compartida con la CPU, evitando copias redundantes de memoria y permitiendo generar tokens de texto y transcribir audio en tiempo real dentro de una envolvente térmica pasiva de apenas $2	ext{–}5	ext{ W}$.

---

## 5. Conclusiones

No existe un acelerador universalmente óptimo para todas las cargas de Inteligencia Artificial; la elección arquitectónica responde al equilibrio entre **intensidad aritmética, presupuesto energético, tamaño de lote (*batch size*) y flexibilidad programática**:

1. **GPUs** dominan cuando se requiere versatilidad de software (CUDA) y alto rendimiento tanto en entrenamiento como en inferencia de modelos complejos o dinámicos.
2. **TPUs** maximizan el rendimiento por vatio y por dólar en entrenamiento e inferencia a gran escala de modelos matriciales densos mediante arreglos sistólicos y compilación estática XLA.
3. **NPUs** son indispensables en el ecosistema móvil y *edge* donde la eficiencia energética (`TOPS/W`) y la cuantización entera (`INT8/INT4`) habilitan IA privada y autónoma.
4. **FPGAs** resuelven nichos críticos de latencia determinista sub-milisegundo (`batch = 1`), integración directa con sensores/redes y tipos de datos personalizados en silicio.
