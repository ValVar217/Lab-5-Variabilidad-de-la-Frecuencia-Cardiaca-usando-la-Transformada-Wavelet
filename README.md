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
  |*Fig 5 : Señal filtrada teniendo en cuenta cada Intervalo R-R.*|   
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
  👆 **Analisis** 👆   
Esta imagen representa el "Análisis de los intervalos R-R en el dominio del tiempo", de los intervalos R-R tras la detección de los picos R. En este gráfico se representan los intervalos de tiempo entre latidos consecutivos en función del tiempo, utilizando una línea azul clara para mostrar cada intervalo R-R detectado, además de una línea discontinua roja que indica la media de estos intervalos. El valor promedio mostrado (1.743 segundos) se traduce en una frecuencia cardíaca promedio baja, de aproximadamente 34.42 bpm, lo cual podría deberse a un estado de reposo profundo, una señal con pocos picos válidos detectados o algún artefacto en la señal. La variabilidad observada entre los intervalos indica fluctuaciones notables en el ritmo cardíaco, lo que puede ser indicativo de una alta variabilidad de la frecuencia cardíaca (HRV), aunque también podría sugerir inconsistencias en la detección de picos dependiendo de la calidad de la señal original. Este tipo de análisis es clave para evaluar la actividad autonómica del corazón en el tiempo.     
  
![WhatsApp Image 2025-05-01 at 6 10 04 PM](https://github.com/user-attachments/assets/f313668e-256e-476e-bffe-37817ad5616c)  
 |*Fig 7 : Parámetros básicos de la HRV en el dominio del tiempo.*|   
   👆 **Analisis** 👆  
Los resultados obtenidos en el análisis de los intervalos R-R en el dominio del tiempo muestran que la media del intervalo entre latidos es de aproximadamente 1.743 segundos, lo cual corresponde a una frecuencia cardíaca promedio de 34.42 latidos por minuto (bpm), lo que podría indicar bradicardia o un estado de relajación profunda. Además, la desviación estándar de 1.123 segundos refleja una variabilidad considerable entre los intervalos R-R, lo que sugiere una alta variabilidad de la frecuencia cardíaca (HRV), posiblemente relacionada con una influencia dominante del sistema nervioso parasimpático o con irregularidades en el ritmo, es decir, a este tipo de comportamiento puede deberse a factores como la respiración, cambios posturales, estado emocional o incluso a distintas condiciones clínicas, y por eso es importante complementar el análisis con herramientas tiempo-frecuencia para observar si estas variaciones siguen un patrón fisiológico coherente o reflejan una disfunción. 
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
![Transf-Wavelet](https://github.com/user-attachments/assets/9d207344-0e3f-4242-9d76-36358b2f7e0d) 
 |*Fig 8 : Resultados --> Transformada de Wavelet.*|          
  👆 **Analisis** 👆     
 La imagen corresponde al análisis espectral de los intervalos R-R mediante la transformada wavelet continua (CWT), específicamente utilizando la wavelet de Morlet. Este gráfico, muestra un espectrograma en el que se observa la evolución temporal de la actividad en distintas bandas de frecuencia, permitiendo visualizar cómo varía la potencia de la señal de los intervalos R-R a lo largo del tiempo. 
- En el eje horizontal se representa el tiempo en segundos, mientras que el eje vertical indica la frecuencia en hertz (Hz), con un límite superior de 0.5 Hz, abarcando así las bandas clásicas del análisis de la variabilidad de la frecuencia cardíaca (HRV): la banda de baja frecuencia (LF), entre 0.04 y 0.15 Hz, relacionada con la actividad simpática y parasimpática, y la banda de alta frecuencia (HF), entre 0.15 y 0.4 Hz, asociada principalmente con la modulación parasimpática.  

Por otro lado, es importante tener en cuenta que la intensidad del color en el gráfico representa la amplitud o potencia de la señal en cada punto temporal y frecuencia. Luego, estan las zonas en tonos amarillos y verdes que indican mayor amplitud, mientras que los tonos azules y morados corresponden a menor actividad. Se destacan patrones periódicos con alta amplitud, especialmente en la banda cercana a los 0.4–0.45 Hz, lo que sugiere una fuerte modulación en la banda de alta frecuencia. Esta actividad sugiere un predominio del control parasimpático sobre la frecuencia cardíaca en ese intervalo de tiempo. La presencia de estos patrones repetitivos y definidos a lo largo del espectrograma refleja una regulación autonómica activa y sostenida del ritmo cardíaco, lo cual puede interpretarse como un signo de un sistema de control cardíaco dinámico y en funcionamiento.

_________________________________        
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
![WhatsApp Image 2025-05-02 at 11 45 34 AM](https://github.com/user-attachments/assets/bf2e2da0-e3d9-45ef-b141-e20d239368a5)  
 |*Fig 9 : Análisis de Frecuencias de HRV (Estimación con CWT).*|        
A partir del espectrograma de los intervalos R-R y el análisis de frecuencias de la variabilidad de la frecuencia cardíaca (HRV) usando la transformada wavelet, se puede decir lo siguiente: Aunque en el espectrograma se nota bastante actividad en la zona de alta frecuencia (HF), los valores calculados muestran que la potencia en la banda de baja frecuencia (LF) es un poco mayor (1.3930) que la de alta frecuencia (0.9205). Esto da un cociente LF/HF de 1.51. Ese valor indica que hay un leve predominio del sistema nervioso simpático (que activa el cuerpo), aunque también hay una buena participación del sistema parasimpático (que relaja el cuerpo). En resumen, el control del corazón está equilibrado, pero con una ligera inclinación hacia la activación.  
_________________________________    

## CONCLUSIONES: ⚙️     
Al finalizar la práctica, se puede concluir que el análisis en el dominio del tiempo, representado por el gráfico de intervalos R-R, permite observar de forma directa la variabilidad entre los latidos del corazón, evidenciando fluctuaciones y posibles irregularidades en los intervalos. Sin embargo, este tipo de análisis no proporciona información sobre cómo varía la frecuencia de estos cambios a lo largo del tiempo. En contraste, el análisis en el dominio tiempo-frecuencia, como el espectrograma generado mediante la Transformada Wavelet Continua (CWT), permite visualizar no solo las frecuencias presentes en la señal, sino también cómo estas evolucionan con el tiempo, revelando patrones oscilatorios que no son evidentes en el dominio únicamente temporal. La elección de la función wavelet influye directamente en la resolución y en la capacidad de detección de ciertos patrones: algunas funciones pueden resaltar mejor las frecuencias bajas o altas, o detectar cambios bruscos, lo que puede modificar ligeramente la interpretación de los resultados, aunque en este caso se utilizó la wavelet de Morlet, que ofrece un buen balance entre resolución temporal y frecuencial. En cuanto a las aplicaciones reales, esta práctica es fundamental para el análisis de la variabilidad de la frecuencia cardíaca (HRV), la cual es una herramienta clínica y de investigación útil en la evaluación del sistema nervioso autónomo, el estrés, la fatiga, y la recuperación en el deporte, así como en el monitoreo de pacientes con enfermedades cardíacas o trastornos del sueño, más especificamente podriamos responder las siguientes preguntas para mayor analisis: 

**1. ¿Qué diferencias se observan entre los análisis en el dominio del tiempo y el dominio tiempo-frecuencia?**  
El análisis en el dominio del tiempo, como el que se realiza al detectar los picos R en la señal ECG, permite observar directamente eventos específicos como latidos cardíacos, amplitudes y morfología de las ondas, siendo útil para detectar arritmias o latidos irregulares. Sin embargo, este tipo de análisis no muestra cómo cambian las características frecuenciales de la señal a lo largo del tiempo. En cambio, el análisis en el dominio tiempo-frecuencia (como el espectrograma generado con la transformada wavelet continua) permite observar la evolución dinámica de las frecuencias asociadas a los intervalos R-R, evidenciando modulaciones y patrones que no se pueden detectar únicamente en el tiempo. Esto es especialmente útil para estudiar la variabilidad del ritmo cardíaco y su regulación por el sistema nervioso autónomo.  

**2. ¿Qué efecto tiene el uso de diferentes funciones wavelet en los resultados del análisis?**    
El tipo de wavelet utilizado afecta directamente la resolución temporal y frecuencial del análisis. Por ejemplo, la wavelet de Morlet, empleada en esta práctica, ofrece un buen equilibrio entre resolución en tiempo y frecuencia, siendo adecuada para estudiar señales fisiológicas como el ECG donde se requiere captar tanto la localización de eventos rápidos como la evolución lenta de frecuencias. Si se utilizara una wavelet como la Haar (más discontinua), se obtendría mejor resolución temporal pero peor resolución frecuencial, lo cual podría dificultar la identificación precisa de las bandas de frecuencia vinculadas a la modulación autonómica. Por tanto, la elección de la wavelet debe hacerse considerando el tipo de señal y la información que se desea extraer.  

**3. ¿Qué aplicaciones reales tiene esta práctica?**
Esta práctica tiene aplicaciones directas en el análisis clínico y biomédico del ritmo cardíaco, en particular para evaluar la variabilidad de la frecuencia cardíaca (HRV), un marcador importante del equilibrio entre los sistemas simpático y parasimpático del sistema nervioso autónomo. El uso combinado del análisis en el tiempo (detección de picos R) y en tiempo-frecuencia (CWT) permite identificar patrones de estrés, fatiga, arritmias, y otras condiciones fisiológicas o patológicas. Además, estas técnicas son fundamentales en dispositivos portátiles como smartwatches o monitores cardíacos, y se usan también en investigación sobre el sueño, rendimiento deportivo, y monitoreo remoto de pacientes.

___________________________________       

## Licencia 
Open Data Commons Attribution License v1.0

## Temas:
# 📡 Procesamiento de Señales  
- Adquisición de la señal ECG en tiempo real durante periodo de tiempo.  
- Aplicación de filtros IIR y Butterworth para eliminar ruido e interferencias no deseadas.    

# 🔊 Análisis en Frecuencia  
- Aplicación de la Transformada de Wavelet.
- Cálculo de la Transformada Wavelet, Intervalos entre cada R (pico) del ECG, media movil.  

# 🖥️ Código e Implementación  
- Explicación del código utilizado para la adquisición, filtrado y análisis de la señal ECG.
- Implementación de gráficos para visualizar la evolución de la frecuencia en el tiempo y la distribución de todo.
- Mejoras en la optimización del código, asegurando una correcta segmentación de los datos y reduciendo errores en el análisis estadístico.
 

