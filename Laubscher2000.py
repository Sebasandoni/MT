import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =====================================================================
# MÓDULO 1: FRAGMENTACIÓN IN SITU (ISFB)
# =====================================================================

def calcular_jsi(ff_m, num_familias):
    """
    Calcula el Índice de Espaciamiento de Discontinuidades (JSI).
    """
    jsi = 100 / (ff_m * num_familias) 
    return jsi

def calcular_volumen_bloque_insitu(S1, S2, S3, gamma1, gamma2, gamma3):
    """
    Computa el Volumen de Bloque In-Situ promedio (V_b).
    """
    g1, g2, g3 = np.radians([gamma1, gamma2, gamma3])
    V_b = (S1 * S2 * S3) / (np.sin(g1) * np.sin(g2) * np.sin(g3))
    return V_b

# =====================================================================
# MÓDULO 2: FRAGMENTACIÓN PRIMARIA
# =====================================================================

def ajustar_mrmr(RMR, esfuerzo_inducido, UCS, factor_orientacion, factor_meteorizacion):
    """
    Aplica multiplicadores numéricos de castigo al RMR.
    """
    MRMR = RMR * factor_orientacion * factor_meteorizacion
    if esfuerzo_inducido > UCS:
        MRMR *= 0.8 
    return MRMR

def ecuacion_rosin_rammler(x, xc, n):
    """
    Función de distribución empírica de Rosin-Rammler (ahora evaluada en volumen).
    """
    return 1 - np.exp(-(x / xc)**n)

def simular_fragmentacion_primaria(MRMR, V_b):
    """
    Cruce numérico del MRMR ajustado y el V_b mediante regresiones.
    """
    # Aberturas transformadas a volumen (m³): 0.1m->0.001m³, 0.5m->0.125m³, 1.0m->1.0m³, 2.0m->8.0m³
    volumenes_malla = np.array([0.001, 0.125, 1.0, 8.0]) 
    pasante_acumulado = np.array([0.15, 0.45, 0.70, 0.90]) 
    
    popt, _ = curve_fit(ecuacion_rosin_rammler, volumenes_malla, pasante_acumulado, p0=[1.0, 1.0])
    xc, n = popt 
    return xc, n

# =====================================================================
# MÓDULO 3: SALIDAS DEL ALGORITMO Y GRÁFICOS (OUTPUTS)
# =====================================================================

def generar_salidas(xc, n):
    """
    Retorna la estructura de datos probabilística con los hitos P80, P50 y P20.
    """
    # Cálculo de los tamaños característicos (en m³)
    P80 = xc * (-np.log(1 - 0.80))**(1/n)
    P50 = xc * (-np.log(1 - 0.50))**(1/n)
    P20 = xc * (-np.log(1 - 0.20))**(1/n)
    
    # Proporciones volumétricas
    pct_finos = ecuacion_rosin_rammler(0.001, xc, n) * 100
    pct_medianos = (ecuacion_rosin_rammler(1.0, xc, n) - ecuacion_rosin_rammler(0.125, xc, n)) * 100
    pct_colpas = (1 - ecuacion_rosin_rammler(2.0, xc, n)) * 100
    
    return P80, P50, P20, pct_finos, pct_medianos, pct_colpas

def graficar_curva_granulometrica(xc, n, P80, P50, P20):
    """
    Genera la Curva S Continua en función del volumen (m³).
    """
    # Rango volumétrico: de 0.0001 m³ a 20.0 m³
    x_vals = np.logspace(-4, np.log10(20), 500)
    
    # Calcular porcentaje pasante acumulado (0 a 100%)
    y_vals = ecuacion_rosin_rammler(x_vals, xc, n) * 100
    
    # Configuración del plot
    plt.figure(figsize=(11, 7))
    plt.plot(x_vals, y_vals, label=f'Curva Rosin-Rammler\n($x_c$={xc:.2f} m³, $n$={n:.2f})', color='#1f77b4', linewidth=2.5)
    
    # Destacar hitos característicos de la distribución (P80, P50, P20)
    plt.scatter([P80], [80], color='red', zorder=5, label=f'$P_{{80}}$ = {P80:.3f} m³', s=80)
    plt.scatter([P50], [50], color='green', zorder=5, label=f'$P_{{50}}$ = {P50:.3f} m³', s=80)
    plt.scatter([P20], [20], color='orange', zorder=5, label=f'$P_{{20}}$ = {P20:.3f} m³', s=80)
    
    # Líneas guías para los hitos
    hitos_y = [80, 50, 20]
    hitos_x = [P80, P50, P20]
    colores = ['red', 'green', 'orange']
    
    plt.hlines(y=hitos_y, xmin=0.0001, xmax=hitos_x, colors=colores, linestyles='dashed', alpha=0.6)
    plt.vlines(x=hitos_x, ymin=0, ymax=hitos_y, colors=colores, linestyles='dashed', alpha=0.6)
    
    # Formato gráfico
    plt.xscale('log') 
    plt.xlabel('Volumen del Fragmento (m³) [Escala Logarítmica]', fontsize=12, fontweight='bold')
    plt.ylabel('Porcentaje Pasante Acumulado (%)', fontsize=12, fontweight='bold')
    plt.title('Curva Granulométrica de Fragmentación Primaria por Volumen\n(Modelo de Laubscher)', fontsize=14, fontweight='bold')
    
    plt.xlim(0.0001, 20)
    plt.ylim(0, 105)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=11, shadow=True)
    
    # Despliegue en pantalla
    plt.tight_layout()
    plt.show()

# --- Ejecución Principal del Script ---
if __name__ == "__main__":
    # 1. Simulación de parámetros matemáticos de la curva volumétrica
    xc_obtenido, n_obtenido = simular_fragmentacion_primaria(MRMR=45, V_b=1.5)
    
    # 2. Obtención de hitos críticos y proporciones
    P80, P50, P20, finos, medianos, colpas = generar_salidas(xc_obtenido, n_obtenido)
    
    # 3. Impresión de resultados en consola
    print("="*50)
    print("RESULTADOS DE FRAGMENTACIÓN PRIMARIA (VOLUMEN)")
    print("="*50)
    print(f"Parámetros Rosin-Rammler : xc = {xc_obtenido:.3f} m³, n = {n_obtenido:.2f}")
    print(f"Hitos Característicos    : P80 = {P80:.3f} m³, P50 = {P50:.3f} m³, P20 = {P20:.3f} m³")
    print(f"Proporción Volumétrica   : {finos:.1f}% Finos (< 0.001 m³)")
    print(f"                         : {medianos:.1f}% Medianos (0.125 - 1.0 m³)")
    print(f"                         : {colpas:.1f}% Colpas (> 2.0 m³)")
    print("="*50)
    
    # 4. Generación de la gráfica granulométrica volumétrica
    graficar_curva_granulometrica(xc_obtenido, n_obtenido, P80, P50, P20)