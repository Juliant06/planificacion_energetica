# Remoción de la estacionalidad
import numpy as np
class lanczos:


    def __init__(
            self, 
            N:int, 
            data:np.array,
            fc_rapida:float,
            fc_lenta:float
        ):

        self.N = N    # Instance attribute
        self.data = data
        self.fc_rapida = fc_rapida
        self.fc_lenta = fc_lenta

    def pesos_lanczos_pasobajo(self,fc):
        """
        Calcula los pesos de un filtro de paso bajo de Lanczos.
        N : Semi-ventana del filtro (meses hacia atrás y hacia adelante)
        fc: Frecuencia de corte (1 / meses)
        """
        # Vector de índices k (desde -N hasta N)
        k = np.arange(-self.N, self.N + 1)
        
        # --- PASO 1: Filtro Ideal (Función Sinc normalizada) ---
        # Ecuación: h_k = sin(2*pi*fc*k) / (pi*k)
        h_k = np.zeros_like(k, dtype=float)
        h_k[k == 0] = 2 * fc  # Límite en k=0
        h_k[k != 0] = np.sin(2 * np.pi * fc * k[k != 0]) / (np.pi * k[k != 0])
        
        # --- PASO 2: La Ventana de Lanczos ---
        # Ecuación: w_k = sin(pi*k/N) / (pi*k/N)
        w_k = np.zeros_like(k, dtype=float)
        w_k[k == 0] = 1.0  # Límite en k=0
        w_k[k != 0] = np.sin(np.pi * k[k != 0] / self.N) / (np.pi * k[k != 0] / self.N)
        
        # --- PASO 3: Calcular los pesos crudos ---
        # Ecuación: w'_k = h_k * w_k
        w_prime_k = h_k * w_k
        
        # --- PASO 4: La corrección de Duchon ---
        # Ecuación: C = (1 - sum(w'_k)) / (2N + 1)
        suma_pesos_crudos = np.sum(w_prime_k)
        C = (1.0 - suma_pesos_crudos) / (2 * self.N + 1)
        
        # Pesos finales normalizados: W_k = w'_k + C
        W_k = w_prime_k + C
        
        return W_k

# =============================================================================
# CREACIÓN DE LA SEÑAL SINTÉTICA 
# =============================================================================

    def filtro_lanczos(self):

        meses = np.arange(len(self.data))

        # 1. Creamos el filtro de paso bajo para frecuencias rápidas
        W_rapido = self.pesos_lanczos_pasobajo(
            self.fc_rapida
        )

        # 2. Creamos el filtro de paso bajo para frecuencias muy lentas
        W_lento = self.pesos_lanczos_pasobajo(
            self.fc_lenta
        )

        # --- resta para crear el pasabanda ---
        W_pasabanda = W_rapido - W_lento

        # --- La Convolución ---
        data_filtrado = np.convolve(
            self.data, 
            W_pasabanda, 
            mode='valid'
        )
 
        return data_filtrado

        # # Para poder graficarlo alineado con la señal original, rellenamos los bordes perdidos con NaN
        # viento_filtrado_alineado = np.empty_like(self.data)
        # viento_filtrado_alineado[:] = np.nan
        # viento_filtrado_alineado[self.N:-self.N] = data_filtrado

