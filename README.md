<h1 align="center"> Lab 5 / Variabilidad de la Frecuencia Cardiaca usando la Transformada Wavelet </h1>    
<h2 align="center">  🫀 Medicion de ECG ❤️ </h2>  

# INTRODUCCIÓN 
Mediante el desarrollo del presente informe, se presenta la realización de la práctica de laboratorio enfocada en el análisis de la variación del ritmo cardíaco (HRV), con el objetivo de comprender cómo responde el sistema nervioso autónomo ante diferentes estímulos fisiológicos. Durante la práctica, se utilizó el software Python junto con sensores de pulso para registrar la señal cardíaca en distintas condiciones como: en reposo, durante una respiración controlada, ejecutando la maniobra de Valsalva y en el periodo de recuperación después del estres. A partir de estas señales se analizaron los intervalos entre latidos consecutivos, conocidos como intervalos R-R, que permiten observar la variabilidad cardíaca. Este parámetro es fundamental para evaluar el equilibrio entre las ramas simpática y parasimpática del sistema nervioso, ya que una mayor variabilidad indica una buena adaptación del organismo, mientras que una baja variabilidad puede estar relacionada con estrés o disfunción autonómica. El propósito de este laboratorio es desarrollar habilidades en la adquisición y análisis de señales biológicas, interpretando la HRV como una herramienta útil para estudiar la actividad del sistema nervioso en tiempo real bajo diversas condiciones fisiológicas.

![Diagrama de Flujo](https://github.com/user-attachments/assets/f439cc32-6c88-4985-9688-d35bbe3017fe)    
  |*Fig 1 : Diagrama de Flujo - Procedimiento de elaboracion de la guia de laboratorio.*| 


<h1 align="center"> 📄 GUIA DE USUARIO 📄 </h1>   

## ✔️ANALISIS Y RESULTADOS  

## a. Fundamento teórico:
**🧠 Sistema Nervioso Autónomo (SNA):** El SNA regula funciones automáticas del cuerpo, como la frecuencia cardíaca, la respiración y la digestión. Se divide en:  
- Simpático: activa el cuerpo ante el estrés (acelera el corazón).  
- Parasimpático: promueve el descanso y la recuperación (ralentiza el corazón).  
- El equilibrio entre ambos se refleja en la variabilidad de la frecuencia cardíaca.

**❤️ Variabilidad de la Frecuencia Cardíaca (HRV)**  
La HRV es la variación en el tiempo entre latidos consecutivos (intervalos R-R en un ECG).
- Alta HRV: indica buena adaptación del sistema cardiovascular y predominio parasimpático.
- Baja HRV: puede reflejar estrés, fatiga o riesgo cardiovascular.
Se puede analizar en el dominio del tiempo (estadísticas simples) o en el dominio de la frecuencia (análisis espectral).

**🌊 Transformada Wavelet**  
Es una herramienta que permite analizar señales no estacionarias (como el ECG) descomponiéndolas en tiempo y frecuencia simultáneamente. A diferencia de la transformada de Fourier, la wavelet puede detectar cambios transitorios, lo que la hace ideal para estudiar cómo varían las frecuencias (LF y HF) de la HRV a lo largo del tiempo.
_________________________________    
## b. Adquisición de la señal ECG: 
Para la obtención de la señal ECG se utilizó un sistema de adquisición de datos DAQ6002, conectado a tres electrodos de superficie colocados según la configuración estándar de derivación Einthoven (brazos derecho, izquierdo y pierna derecha como referencia o tierra). La señal fue registrada durante 5 minutos en condiciones de reposo, con el sujeto sentado de manera relajada y respirando de forma natural, en un ambiente controlado para minimizar interferencias eléctricas. La señal se muestreó a una frecuencia de [1000 Hz], lo cual permite una adecuada resolución temporal para el análisis de los complejos QRS. Los datos fueron almacenados y posteriormente procesados en Python para el análisis de la variabilidad de la frecuencia cardíaca (HRV).  
   
--> En esta primera parte del código se realiza la carga e inicialización de los datos de la señal ECG. Primero se importan las librerías necesarias: pandas para manejar archivos de Excel y estructuras de datos tipo DataFrame; matplotlib.pyplot para realizar gráficos; numpy para cálculos numéricos eficientes; scipy.signal para procesamiento de señales, y pywt para aplicar transformadas wavelet. Luego, se especifica la ruta del archivo Excel que contiene la señal ECG y se carga en un DataFrame usando pd.read_excel(). A partir de este archivo, se extraen dos columnas: la primera (df.iloc[:, 0].values) representa el tiempo en milisegundos o segundos, y la segunda (df.iloc[:, 1].values) contiene la señal cruda del ECG, es decir, los valores eléctricos medidos desde el corazón. Finalmente, se define la frecuencia de muestreo (fs = 1000), lo que indica que la señal fue registrada a mil muestras por segundo. Esto será esencial más adelante para convertir índices de muestras en tiempo real, calcular frecuencias y diseñar filtros:

```python  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt
import pywt.data

# Cargar datos
file_path = r"C:\Users\Esteban\Downloads\ECG_organizado.xlsx"
df = pd.read_excel(file_path)
t = df.iloc[:, 0].values
ecg_signal_raw = df.iloc[:, 1].values
fs = 1000  # Hz
```
![Señal original](https://github.com/user-attachments/assets/6b3ff343-5bc2-4300-b4f3-df3fed83c966)  
  |*Fig 2 : Señal original.*|       
👆 **Analisis** 👆  
La señal mostrada en la imagen corresponde a la ECG que al momento se encuentra sin procesar, tal como fue adquirida directamente del sistema de adquisición (DAQ6002). En ella se puede observar un alto nivel de ruido, con fluctuaciones rápidas y amplitudes que oscilan aproximadamente entre 1 y 5 mV, lo cual no es típico de una señal ECG fisiológica normal. Esta distorsión puede deberse a interferencias eléctricas, artefactos por movimiento o actividad muscular, lo que impide que podamos identificar con claridad los componentes característicos del ciclo cardíaco en este caso, como los complejos QRS. Esta visualización inicial requiere la necesidad de aplicar técnicas de preprocesamiento, como el filtrado pasabanda (Butterworth) y la filtración por mediana, que serán implementadas en las etapas siguientes del código para limpiar la señal y permitir una detección precisa de los picos R.
_________________________________    
Ahora bien, En esta siguiente sección del código se realiza el preprocesamiento de la señal ECG, lo cual es esencial para eliminar ruidos y preparar la señal para un análisis más preciso.

## c. Pre - procesamiento de la señal:      
▪️ **1. Filtro IIR Butterworth**  
Primero se diseña un filtro pasa banda Butterworth de cuarto orden para dejar pasar únicamente las frecuencias relevantes del ECG, que suelen estar entre 0.5 Hz y 35 Hz. Estos valores se eligen porque:
- El componente útil del ECG, como las ondas P, QRS y T, se encuentra en ese rango.  
- Frecuencias menores pueden contener ruido de movimientos (muy lento), y mayores a 35 Hz pueden incluir interferencias de alta frecuencia o del ambiente.
--> A demas, usando la función predefinida "scipy.signal.bilinear" para obtener la transformación bilineal de una función de transferencia analógica, se obtuvo los coeficientes de entrada (b = [ 1.1951712e-06, -1.1923701e-06, -1.1923701e-06,  1.1951712e-06 ]) y los coeficientes de salida (a = [ 1.0, -2.98538391, 2.97112673, -0.98574185 ]) quedando así la ecuación en diferencia de la siguiente manera:  

![WhatsApp Image 2025-05-02 at 11 36 18 PM](https://github.com/user-attachments/assets/8335b9b5-ad89-410a-8bf9-163b389ec603)    
  |*Fig 3 : Diseño del filtro   IIR.*| 

Y bueno, para implementar el filtro, se calcula la frecuencia de Nyquist (nyq), que es la mitad de la frecuencia de muestreo (fs/2). Este es utilizado para limpiar señales ECG eliminando componentes de baja frecuencia (ruido por movimiento) y de alta frecuencia (ruido muscular o eléctrico). Se plantea el filtro con frecuencia de muestreo de 1000 Hz, corte inferior en 0.5 Hz y superior en 35 Hz, a su vez, se realiza una conversión de frecuencias normalizadas a rad/muestra, seguida del pre-warping para transformar las frecuencias digitales en frecuencias analógicas antes de aplicar la transformación bilineal. Tambien, se calcula el orden del filtro usando la fórmula clásica de Butterworth, considerando las atenuaciones deseadas (20 dB fuera de banda y 3 dB dentro), lo que lleva a un orden n ≈ 4. Finalmente, se incluye la transformación a la frecuencia central Ωc, el polinomio del denominador normalizado del filtro analógico y la función de transferencia digital resultante. Esta última se convierte en una ecuación en diferencias lista para ser implementada, permitiendo el filtrado digital en tiempo discreto sobre la señal original mostrada previamente.

▪️ **2. Filtro mediana**  
Después, se aplica un filtro de mediana a la señal ya filtrada, teniendo en cuenta que este tipo de filtro es útil para eliminar picos abruptos o artefactos que no representan actividad cardíaca real, como pequeñas interferencias y funciona reemplazando cada valor por la mediana de sus "vecinos" (en este caso, de una ventana de 5 muestras), lo que ayuda a suavizar la señal sin deformar las formas de onda importantes del ECG; Pues esto termina de limpiar la señal para facilitar una detección precisa de los picos R en pasos posteriores.   

```python  
# ----------------------------
# 1. Filtro IIR Butterworth
# ----------------------------
lowcut = 0.5
highcut = 35
order = 4
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq

b, a = signal.butter(order, [low, high], btype='band')
filtered_ecg = signal.filtfilt(b, a, ecg_signal_raw)

# ----------------------------
# 2. Filtro mediana
# ----------------------------
median_filtered_ecg = signal.medfilt(filtered_ecg, kernel_size=5)
```
![Señal filtrada](https://github.com/user-attachments/assets/e8bddc2a-eb36-4fa9-a3a3-6e24b0ad4c03)    
  |*Fig 4 : Señal filtrada.*|     
  👆 **Analisis** 👆  
Esta gráfica muestra el resultado de aplicar el filtro IIR Butterworth diseñado previamente a la señal original ECG. A diferencia del gráfico original, donde el ruido era muy grande y saturaba el rango de ±5 mV, aquí la señal está claramente delimitada en un rango más estrecho, siendo más limpia y menos contaminada por frecuencias fuera del rango fisiológico del ECG. Tambien hay que tener en cuen ta que el filtro ha atenuado de forma efectiva las componentes de muy baja frecuencia (movimiento de línea base) y de alta frecuencia (ruido muscular y eléctrico), conservando las componentes principales del ECG, como los complejos QRS, por lo que podemos decir que, la morfología ahora es más identificable y apta para análisis clínico o procesamiento automático. 
_________________________________      
Posterior a lo anterior, se realiza un suavizado adicional de la señal ECG utilizando un filtro de promedio móvil, con el objetivo de reducir pequeñas oscilaciones residuales y dejar la señal más limpia para la detección de eventos cardíacos importantes.      

Primero, se define el tamaño de la ventana de suavizado como 0.03 * fs, es decir, 30 milisegundos de duración. Como la señal está muestreada a 1000 Hz (fs = 1000), eso equivale a 30 muestras. Luego, se aplica el suavizado con np.convolve(), una operación que recorre la señal y calcula el promedio de los valores dentro de esa ventana. Aquí se usa np.ones(window_size)/window_size para crear un filtro promedio, lo que significa que todos los valores dentro de la ventana tienen el mismo "peso" y por ultimo, La opción mode='same' asegura que la salida tenga el mismo tamaño que la señal original. Este suavizado actúa como un “pulido” final para la señal ya filtrada, de esta manera eliminando pequeñas variaciones que podrían confundir al algoritmo de detección de picos R (latidos) más adelante.     

```python  
# ----------------------------
# 3. Suavizado adicional
# ----------------------------
window_size = int(0.03 * fs)  # 30 ms
smoothed_ecg = np.convolve(median_filtered_ecg, np.ones(window_size)/window_size, mode='same')
```  
Siguiendo con el codigo, tenemos:   
▪️ **Detección de picos R**   
Aquí se utiliza la función signal.find_peaks() de SciPy para identificar los picos prominentes de la señal suavizada, que corresponden a los picos R del ECG (los eventos más sobresalientes en cada ciclo cardíaco).

- El parámetro distance se fija en int(0.6 * fs), pues es un intervalo mínimo de 600 milisegundos (0.6 segundos) entre picos, y en parte es equivalente a una frecuencia cardíaca máxima de 100 latidos por minuto, pero ¿Que cual es su funcion? pues, este valor actúa como filtro para evitar detectar falsos positivos (picos que no son R reales).  
- El parámetro height=np.max(smoothed_ecg)*0.35 indica que solo se considerarán picos con una altura mínima del 35% del valor máximo de la señal suavizaday esto se hace con el finde que esto descarta pequeños picos de ruido o de otras ondas (como T o P) que no corresponden al complejo QRS; y como resultado es un array peaks que contiene los índices de los picos R detectados en la señal temporal.

▪️**5. Cálculo de intervalos R-R**
Una vez localizados los picos R, se calculan los intervalos R-R como la diferencia entre los tiempos de cada pico consecutivo, usando np.diff(t[peaks]). Estos intervalos representan el tiempo entre un latido y el siguiente, medidos en segundos. Tambien, se usa np.diff() para calcular la diferencia entre elementos consecutivos del array t[peaks], que contiene los tiempos asociados a cada pico R. Además, se incluye una verificación con if len(peaks) > 1 para asegurarse de que al menos haya dos picos y así poder calcular al menos un intervalo. Si no se encuentran suficientes picos, se muestra un mensaje y se asigna un array vacío a rr_intervals.

```python  
# ----------------------------
# 4. Detección de picos R
# ----------------------------
# Ajusta el parámetro 'distance' para evitar detectar picos falsos
min_rr_distance = int(0.6 * fs)  # intervalo mínimo R-R = 600 ms (100 bpm)
peaks, _ = signal.find_peaks(smoothed_ecg, distance=min_rr_distance, height=np.max(smoothed_ecg)*0.35)

# ----------------------------
# 5. Calcular intervalos R-R
# ----------------------------
if len(peaks) > 1:  # Asegurarse de que haya al menos dos picos para calcular intervalos
    rr_intervals = np.diff(t[peaks])  # en segundos
else:
    rr_intervals = np.array([])
    print("No se encontraron suficientes picos R para calcular los intervalos R-R.")
```
_________________________________    
## d. Análisis de la HRV en el dominio del tiempo:    
▪️ **6. Crear señal con información de picos R**    
Para la siguiente parte, se construye un nuevo arreglo llamado r_peak_signal, que tiene la misma longitud que la señal suavizada pero está relleno con ceros. Luego, en los índices correspondientes a los picos R detectados, se colocan los valores reales de la señal. Pues, este paso permite crear una señal que solo contiene los picos R, facilitando su análisis individual o la visualización superpuesta. No altera la señal original, sino que crea una representación enmascarada donde los únicos puntos no nulos son los picos detectados. 

▪️**7. Visualización principal con picos R**  
Aquí se crea una gráfica en la que se muestra:
- La señal ECG suavizada en azul ('royalblue'), que representa la actividad cardíaca ya filtrada y limpia. Sobre ella, se superponen los picos R con puntos rojos ('ro'), permitiendo visualizar en qué momentos ocurren los latidos cardíacos.

```python  
# ----------------------------
# 6. Crear señal con información de picos R
# ----------------------------
r_peak_signal = np.zeros_like(smoothed_ecg)
r_peak_signal[peaks] = smoothed_ecg[peaks]

# ----------------------------
# 7. Visualización principal con picos R
# ----------------------------
plt.figure(figsize=(18, 6))
plt.plot(t, smoothed_ecg, label='ECG filtrada', color='royalblue')
plt.plot(t[peaks], smoothed_ecg[peaks], 'ro', label='Picos R Detectados')
plt.title('Detección de Picos R en ECG')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (mV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
![Picos R en ECG](https://github.com/user-attachments/assets/dcda920f-1b8d-420d-bcc5-b82681eab18e)  
  |*Fig 5 : .*|   
La señal obtenida tras el filtrado muestra una forma de onda de ECG mucho más limpia y definida, en la que se distinguen claramente los complejos QRS como picos prominentes y regulares a lo largo del tiempo; sobre esta señal, los puntos rojos indican los picos R detectados, los cuales se alinean con los máximos de cada complejo, evidenciando un ritmo cardíaco estable y una detección exitosa de cada latido. (Identificamos los picos R y calculamos los intervalos R-R  usando la función predefinida  "scipy.signal.find_peaks")  
![Intervalos RR](https://github.com/user-attachments/assets/7c0bd066-efe8-4b81-81e7-fa13aa9d21e4)  
_________________________________   
Bien, el siguiente segmento del código se dedica al análisis de los intervalos R-R en el dominio del tiempo, es decir, al estudio de la variabilidad del ritmo cardíaco (HRV) a partir de los tiempos entre latidos consecutivos, extraídos de los picos R previamente detectados.    
Este análisis en el dominio del tiempo es esencial en el estudio del ritmo cardíaco porque permite:  
- Se grafica cada intervalo R-R en función del tiempo, lo cual permite observar cómo varían los latidos con el tiempo.
- Detectar patrones de bradicardia (frecuencia muy baja) o taquicardia (frecuencia alta).  
- Evaluar la variabilidad de la frecuencia cardíaca (HRV), que es un indicador importante del estado del sistema nervioso autónomo y de la salud cardiovascular.  
- Identificar posibles irregularidades o arritmias.  
Es decir, este bloque proporciona una caracterización cuantitativa y visual de la dinámica cardíaca a partir de los intervalos R-R, facilitando la interpretación clínica o científica del comportamiento del corazón.  

```python  
# ----------------------------
# 8. Análisis de los intervalos R-R en el dominio del tiempo
# ----------------------------
if len(rr_intervals) > 0:
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    heart_rate_mean = 60 / mean_rr
    print("\n--- Análisis de los Intervalos R-R en el Dominio del Tiempo ---")
    print(f"Media de los Intervalos R-R: {mean_rr:.3f} segundos")
    print(f"Desviación Estándar de los Intervalos R-R: {std_rr:.3f} segundos")
    print(f"Frecuencia Cardíaca Promedio (basada en la media de R-R): {heart_rate_mean:.2f} bpm")
    print(f"Variabilidad de la Frecuencia Cardíaca (SDNN): {std_rr * 1000:.2f} ms")

    # Visualización de los intervalos R-R
    time_points_rr = t[peaks[:-1]]
    plt.figure(figsize=(12, 6))
    plt.plot(time_points_rr, rr_intervals, marker='o', linestyle='-', color='skyblue', label='Intervalos R-R')
    plt.axhline(y=mean_rr, color='red', linestyle='--', label=f'Media R-R: {mean_rr:.3f} s')
    plt.axhline(y=mean_rr + std_rr, color='green', linestyle=':', label=f'Media + Desv. Estándar: {mean_rr + std_rr:.3f} s')
    plt.axhline(y=mean_rr - std_rr, color='green', linestyle=':', label=f'Media - Desv. Estándar: {mean_rr - std_rr:.3f} s')
    plt.xlabel('Tiempo del Pico R Anterior (s)')
    plt.ylabel('Intervalo R-R (s)')
    plt.title('Análisis de Intervalos R-R en el Tiempo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
```
![WhatsApp Image 2025-05-01 at 6 08 58 PM](https://github.com/user-attachments/assets/a7d95c33-c197-48bd-b996-834efbf55fd5)
  |*Fig 6 : Grafico creado a partir de los intervalos hallados entre los R-R (HRV).*|    
![WhatsApp Image 2025-05-01 at 6 10 04 PM](https://github.com/user-attachments/assets/f313668e-256e-476e-bffe-37817ad5616c)
 |*Fig 7 : Parámetros básicos de la HRV en el dominio del tiempo.*|     
_________________________________    
## e. Aplicación de transformada Wavelet:       
Se realiza un análisis espectral de la variabilidad de la frecuencia cardíaca (HRV) en el dominio tiempo-frecuencia mediante la Transformada Wavelet Continua (CWT).   Evaluar cómo varía la energía (amplitud) en distintas bandas de frecuencia del ritmo cardíaco a lo largo del tiempo. Esto es útil para poder identificar la actividad del sistema nervioso simpático y parasimpático y tambien, poder analizar la HRV en condiciones de no estacionariedad, algo en lo que las wavelets sobresalen frente al análisis de Fourier que hemos trabajado anteriormente.  
Y teniendo en cuenta que se asume una frecuencia de muestreo constante de 1 Hz sobre la serie de intervalos R-R, lo cual es una simplificación válida si los intervalos están más o menos espacioados significativamente.   
Todo esto, mostrando un espectrograma con amplitud de cada frecuencia en cada instante de tiempo. 

Es importante aclarar que este espectrograma con CWT es una herramienta para poder observar cómo varía la HRV en el tiempo con mucho más detalle que el análisis de frecuencia clásico, a parte es útil en estudios médicos, de estrés o de calidad del sueño.    

```python  
    # ----------------------------
    # 9. Espectrograma de los Intervalos R-R usando Wavelet Continua
    # ----------------------------
    low_freq = 0.04
    high_freq = 0.4
    sampling_rate_rr = 1.0  # Intenta con una frecuencia de muestreo fija

    wavelet = 'morl'
    widths = np.arange(1, 64)

    cwtmatr, freqs = pywt.cwt(rr_intervals - np.mean(rr_intervals), widths, wavelet, sampling_rate_rr)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(rr_intervals) / sampling_rate_rr, freqs[-1], freqs[0]], aspect='auto',
               cmap='viridis', vmin=0, vmax=np.percentile(np.abs(cwtmatr), 95))
    plt.colorbar(label='Amplitud')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.title('Espectrograma de los Intervalos R-R (CWT - Morlet)')
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.show()
```
Y para la ultima parte de nuestro codigo, se realizo el análisis espectral de la variabilidad de la frecuencia cardíaca (HRV) usando la transformada wavelet continua (CWT), enfocándose específicamente en dos bandas fisiológicas importantes: la banda de baja frecuencia (LF: 0.04–0.15 Hz) y la de alta frecuencia (HF: 0.15–0.4 Hz). Se crean máscaras lógicas (lf_mask y hf_mask) para seleccionar los coeficientes de CWT que caen dentro de esas bandas, y luego se calcula la potencia promedio de cada banda como el valor cuadrático medio de los coeficientes en esas frecuencias. Finalmente, se imprime la potencia de ambas bandas y se calcula el índice LF/HF, que es un indicador clásico del balance entre actividad simpática y parasimpática en el sistema nervioso autónomo. También se imprimen los intervalos R-R para referencia adicional del análisis temporal.

```python  
    # Análisis en la banda de baja frecuencia (LF) y alta frecuencia (HF)
    lf_mask = (freqs >= low_freq) & (freqs <= 0.15)
    hf_mask = (freqs > 0.15) & (freqs <= high_freq)

    lf_power = np.mean(np.abs(cwtmatr[lf_mask, :])**2)
    hf_power = np.mean(np.abs(cwtmatr[hf_mask, :])**2)

    print("\n--- Análisis de Frecuencias de HRV (Estimación con CWT) ---")
    print(f"Potencia en la Banda de Baja Frecuencia (LF: 0.04 - 0.15 Hz): {lf_power:.4f}")
    print(f"Potencia en la Banda de Alta Frecuencia (HF: 0.15 - 0.4 Hz): {hf_power:.4f}")
    if hf_power > 0:
        lf_hf_ratio = lf_power / hf_power
        print(f"Ratio LF/HF: {lf_hf_ratio:.2f}")
    else:
        print("No se pudo calcular el ratio LF/HF (HF power es cero).")

else:
    print("\nNo se encontraron suficientes picos R para realizar el análisis espectral.")

# ----------------------------
# 10. Mostrar intervalos R-R en consola
# ----------------------------
print("\nIntervalos R-R (s):", rr_intervals)
```

## CONCLUSIONES: ⚙️    
 

