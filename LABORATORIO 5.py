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

# ----------------------------
# 3. Suavizado adicional
# ----------------------------
window_size = int(0.03 * fs)  # 30 ms
smoothed_ecg = np.convolve(median_filtered_ecg, np.ones(window_size)/window_size, mode='same')

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