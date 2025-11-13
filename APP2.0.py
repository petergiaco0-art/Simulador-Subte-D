import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 0. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Subte Línea D", layout="wide")

# --- URL de Datos ---
DATA_URL = "https://github.com/petergiaco0-art/Simulador-Subte-D/releases/download/v1.0/202501_D.csv" 

# --- (NUEVO) Parámetros de Calentamiento ---
TIEMPO_CALENTAMIENTO_SEG = 3600 # 1 HORA (3600s) para calentar
TIEMPO_MEDICION_SEG = 3600     # 1 HORA (3600s) para medir
TIEMPO_TOTAL_SIM = TIEMPO_CALENTAMIENTO_SEG + TIEMPO_MEDICION_SEG # 7200s

INTERVALO_LAMBDA_SEG = 900 # 15 min

# --- 1. FUNCIONES DE CARGA Y DATOS ---
@st.cache_data
def load_and_clean_data(url_de_datos):
    st.info(f"Descargando y procesando datos de demanda (Enero)...")
    # ... (El código de carga, limpieza y cálculo de Lambda es IDÉNTICO a la v5) ...
    try:
        df = pd.read_csv(
            url_de_datos, sep=';', encoding='utf-8-sig',
            quoting=csv.QUOTE_NONE, low_memory=False
        )
    except Exception as e:
        st.error(f"Error fatal al leer el CSV desde la URL: {e}")
        return None
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    required_cols = ['FECHA', 'pax_TOTAL', 'ESTACION', 'LINEA', 'MOLINETE', 'DESDE']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: El CSV no contiene todas las columnas requeridas: {required_cols}")
        return None
    df['FECHA'] = df['FECHA'].astype(str).str.replace('"', '', regex=False).str.strip()
    df['pax_TOTAL'] = df['pax_TOTAL'].astype(str).str.replace('"', '', regex=False).str.strip()
    df['LINEA'] = df['LINEA'].astype(str).str.strip()
    df['ESTACION'] = df['ESTACION'].astype(str).str.strip()
    df_linea_d = df[df['LINEA'] == 'LineaD'].copy()
    df_linea_d['ESTACION'] = df_linea_d['ESTACION'].replace('9 de julio', '9 de Julio')
    df_linea_d['ESTACION'] = df_linea_d['ESTACION'].replace('AgÃ¼ero', 'Agüero')
    df_linea_d['FECHA'] = pd.to_datetime(df_linea_d['FECHA'], format='%d/%m/%Y')
    df_linea_d['pax_TOTAL'] = pd.to_numeric(df_linea_d['pax_TOTAL'], errors='coerce')
    df_linea_d.dropna(subset=['pax_TOTAL'], inplace=True)
    df_linea_d['pax_TOTAL'] = df_linea_d['pax_TOTAL'].astype(int)
    print("Iniciando cálculo de Lambda...")
    df_linea_d['DIRECCION'] = df_linea_d.apply(asignar_direccion, axis=1)
    df_con_direccion = df_linea_d[df_linea_d['DIRECCION'] != 'Ponderar'].copy()
    df_para_ponderar = df_linea_d[df_linea_d['DIRECCION'] == 'Ponderar'].copy()
    df_ponderar_catedral = df_para_ponderar.copy()
    df_ponderar_congreso = df_para_ponderar.copy()
    df_ponderar_catedral['DIRECCION'] = 'Hacia Catedral'
    df_ponderar_catedral['pax_TOTAL'] = df_ponderar_catedral['pax_TOTAL'] * 0.5
    df_ponderar_congreso['DIRECCION'] = 'Hacia Congreso'
    df_ponderar_congreso['pax_TOTAL'] = df_ponderar_congreso['pax_TOTAL'] * 0.5
    df_direccional = pd.concat([df_con_direccion, df_ponderar_catedral, df_ponderar_congreso], ignore_index=True)
    df_direccional['pax_TOTAL'] = pd.to_numeric(df_direccional['pax_TOTAL'], errors='coerce')
    df_direccional.dropna(subset=['pax_TOTAL'], inplace=True)
    condicion_pico_1hr = (df_direccional['DESDE'] >= '08:00:00') & (df_direccional['DESDE'] < '09:00:00')
    df_pico_direccional = df_direccional[condicion_pico_1hr].copy()
    df_lambda_final = df_pico_direccional.groupby(
        ['ESTACION', 'DIRECCION', 'DESDE']
    )['pax_TOTAL'].mean().reset_index()
    df_lambda_final = df_lambda_final.rename(columns={'pax_TOTAL': 'pasajeros_PROMEDIO_lambda'})
    print("Cálculo de Lambda completado.")
    return df_lambda_final
def asignar_direccion(row):
    estacion = row['ESTACION']
    molinete = str(row['MOLINETE'])
    if estacion == 'Catedral': return 'Hacia Congreso'
    if estacion == 'Congreso de Tucuman': return 'Hacia Catedral'
    if estacion == '9 de Julio': return 'Ponderar'
    if '_N_' in molinete: return 'Hacia Congreso'
    if '_S_' in molinete or '_SO_' in molinete: return 'Hacia Catedral'
    if '_Este_' in molinete or '_E_' in molinete: return 'Hacia Catedral'
    if '_Oeste_' in molinete or '_O_' in molinete: return 'Hacia Congreso'
    return 'Ponderar'

# --- 2. COMPONENTES DEL MODELO SIMPY ---
LISTA_ESTACIONES_NORTE = [
    'Catedral', '9 de Julio', 'Tribunales', 'Callao', 'Facultad de Medicina', 
    'Agüero', 'Pueyrredon.D', 'Bulnes', 'Plaza Italia', 'Ministro Carranza', 
    'Olleros', 'Jose Hernandez', 'Juramento', 'Congreso de Tucuman'
]
LISTA_ESTACIONES_SUR = LISTA_ESTACIONES_NORTE[::-1]
MAPAS_DESCENSO_BASE = {
    "Hacia Catedral": {
        'Congreso de Tucuman': 0.0, 'Juramento': 0.05, 'Jose Hernandez': 0.05, 'Olleros': 0.05,
        'Ministro Carranza': 0.20, 'Plaza Italia': 0.05, 'Bulnes': 0.05,
        'Pueyrredon.D': 0.25, 'Agüero': 0.10, 'Facultad de Medicina': 0.15, 'Callao': 0.20,
        'Tribunales': 0.30, '9 de Julio': 0.40, 'Catedral': 1.0
    },
    "Hacia Congreso": {
        'Catedral': 0.0, '9 de Julio': 0.30, 'Tribunales': 0.25, 'Callao': 0.25,
        'Facultad de Medicina': 0.20, 'Agüero': 0.20, 'Pueyrredon.D': 0.35,
        'Bulnes': 0.25, 'Plaza Italia': 0.35, 'Ministro Carranza': 0.35,
        'Olleros': 0.25, 'Jose Hernandez': 0.25, 'Juramento': 0.30,
        'Congreso de Tucuman': 1.0
    }
}

# --- (MODIFICACIÓN: Clase Anden ahora sabe sobre el calentamiento) ---
class Anden:
    def __init__(self, env, nombre_estacion, direccion, tiempo_calentamiento):
        self.env = env
        self.nombre = f"{nombre_estacion}_{direccion}"
        self.direccion = direccion
        self.estacion_nombre = nombre_estacion
        self.tiempo_calentamiento = tiempo_calentamiento # <-- NUEVO
        self.cola_pasajeros = simpy.Store(env)
        self.metricas_tiempo_espera = []
        self.metricas_cola_por_tren = []

    def add_pasajero(self, pasajero):
        pasajero['tiempo_llegada_anden'] = self.env.now
        return self.cola_pasajeros.put(pasajero)

    def tren_llega_a_anden(self, tren, dwell_time):
        # (NUEVO) Solo registrar métricas DESPUÉS del calentamiento
        # (env.now >= self.tiempo_calentamiento)
        
        # Primero, calculamos cuántos hay en cola (siempre, para el log)
        pasajeros_en_cola_al_llegar = len(self.cola_pasajeros.items)
        # Pero solo guardamos la métrica si estamos midiendo
        if self.env.now >= self.tiempo_calentamiento:
            self.metricas_cola_por_tren.append(pasajeros_en_cola_al_llegar)
        
        pasajeros_subidos = 0
        while tren['pasajeros_actuales'] < tren['capacidad'] and len(self.cola_pasajeros.items) > 0:
            pasajero = yield self.cola_pasajeros.get()
            
            # (NUEVO) Solo registrar métricas DESPUÉS del calentamiento
            if self.env.now >= self.tiempo_calentamiento:
                tiempo_espera = self.env.now - pasajero['tiempo_llegada_anden']
                self.metricas_tiempo_espera.append(tiempo_espera)
            
            tren['pasajeros_actuales'] += 1
            pasajeros_subidos += 1
        
        yield self.env.timeout(dwell_time)

# --- (MODIFICACIÓN: generador_pasajeros ahora corre en bucle) ---
def generador_pasajeros(env, anden, df_lambda_anden, intervalo_seg):
    """
    Este proceso genera pasajeros en bucle, repitiendo el
    horario de 1 hora (4 intervalos) durante toda la simulación.
    """
    intervalos = ['08:00:00', '08:15:00', '08:30:00', '08:45:00']
    
    # Bucle infinito para que el generador nunca se detenga
    while True:
        # Repite el horario de 4 intervalos (1 hora)
        for intervalo_hora in intervalos:
            lambda_actual = 0.0
            try: 
                lambda_actual = df_lambda_anden.loc[intervalo_hora]['pasajeros_PROMEDIO_lambda']
            except KeyError: 
                pass # No hay demanda para este andén en este intervalo
            
            if lambda_actual > 0:
                # Tasa de llegada (pasajeros por segundo)
                tasa_llegada_por_seg = lambda_actual / intervalo_seg
                
                # Bucle para este intervalo de 15 min
                inicio_intervalo = env.now
                while env.now < inicio_intervalo + intervalo_seg:
                    # Tiempo hasta el próximo pasajero
                    tiempo_llegada_prox_pax = random.expovariate(tasa_llegada_por_seg)
                    
                    yield env.timeout(tiempo_llegada_prox_pax)
                    
                    if env.now < inicio_intervalo + intervalo_seg:
                        pasajero = {'id': f"pax_{random.randint(1000,9999)}"}
                        yield anden.add_pasajero(pasajero)
            else:
                # Si no hay lambda, solo esperar 15 min
                yield env.timeout(intervalo_seg)


# (proceso_tren es idéntico a v5)
def proceso_tren(env, tren_id, lista_recorrido, tiempo_viaje, dwell_time, capacidad, mundo_andenes, tasas_descenso):
    tren = {'id': tren_id, 'capacidad': capacidad, 'pasajeros_actuales': 0}
    direccion_viaje = "Hacia Congreso" if lista_recorrido == LISTA_ESTACIONES_NORTE else "Hacia Catedral"
    for i, nombre_estacion in enumerate(lista_recorrido):
        if i > 0:
            yield env.timeout(tiempo_viaje)
        nombre_anden = f"{nombre_estacion}_{direccion_viaje}"
        anden_actual = mundo_andenes[nombre_anden]
        if tren['pasajeros_actuales'] > 0:
            tasa_bajada = tasas_descenso[nombre_estacion]
            pasajeros_que_bajan = np.random.binomial(tren['pasajeros_actuales'], tasa_bajada)
            tren['pasajeros_actuales'] -= pasajeros_que_bajan
        yield env.process(anden_actual.tren_llega_a_anden(tren, dwell_time))

# (MODIFICACIÓN: generador_trenes ahora es más simple)
def generador_trenes(env, direccion, lista_recorrido, min_freq, max_freq, tiempo_viaje, dwell_time, capacidad, mundo_andenes, tasas_descenso):
    """
    Crea un nuevo 'proceso_tren' en la terminal
    cada X minutos (frecuencia estocástica) en un bucle infinito.
    """
    tren_id_counter = 0
    # Bucle infinito para que el generador nunca se detenga
    while True:
        frecuencia = random.uniform(min_freq, max_freq)
        yield env.timeout(frecuencia)
        
        tren_id_counter += 1
        tren_id = f"Tren_{direccion}_{tren_id_counter}"
        env.process(proceso_tren(env, tren_id, lista_recorrido, tiempo_viaje, dwell_time, capacidad, mundo_andenes, tasas_descenso))


# --- 3. FUNCIÓN PRINCIPAL DE SIMULACIÓN (MONTECARLO) ---

def confidence_interval(data):
    if len(data) < 2:
        return (0, 0)
    return stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data))

# --- (MODIFICACIÓN: Función Montecarlo ahora usa CALENTAMIENTO) ---
def run_montecarlo_simulation(
    n_replicaciones, min_freq, max_freq, capacidad, dwell_time, tiempo_viaje, 
    df_lambda, mapas_descenso_base, 
    multiplicador_catedral, multiplicador_congreso, multiplicador_demanda
    ):
    
    print(f"Iniciando Montecarlo con {n_replicaciones} réplicas...")
    lista_resultados_andenes = []
    lista_kpis_globales = []
    
    mapas_descenso_ajustados = {
        "Hacia Catedral": {
            estacion: min(tasa * multiplicador_catedral, 1.0) 
            for estacion, tasa in mapas_descenso_base["Hacia Catedral"].items()
        },
        "Hacia Congreso": {
            estacion: min(tasa * multiplicador_congreso, 1.0) 
            for estacion, tasa in mapas_descenso_base["Hacia Congreso"].items()
        }
    }
    # (NUEVO) Aplicar multiplicador de demanda al DataFrame de Lambda
    df_lambda_ajustado = df_lambda.copy()
    df_lambda_ajustado['pasajeros_PROMEDIO_lambda'] = df_lambda_ajustado['pasajeros_PROMEDIO_lambda'] * multiplicador_demanda
    print(f"Multiplicador de demanda aplicado: {multiplicador_demanda:.1f}x")
    progress_bar = st.progress(0, text="Iniciando simulación...")

    # Bucle de Replicación
    for i in range(n_replicaciones):
        semilla_actual = i + 1
        progress_bar.progress((i + 1) / n_replicaciones, text=f"Corriendo Réplica {semilla_actual}/{n_replicaciones} (Calentando...)")
        
        random.seed(semilla_actual)
        np.random.seed(semilla_actual)
        
        env = simpy.Environment()
        mundo_andenes = {}
        # (NUEVO) Pasar el tiempo de calentamiento a los andenes
        for estacion in LISTA_ESTACIONES_NORTE:
            anden_norte = Anden(env, estacion, "Hacia Congreso", TIEMPO_CALENTAMIENTO_SEG) 
            mundo_andenes[anden_norte.nombre] = anden_norte
            anden_sur = Anden(env, estacion, "Hacia Catedral", TIEMPO_CALENTAMIENTO_SEG) 
            mundo_andenes[anden_sur.nombre] = anden_sur
            
        # (NUEVO) Pre-filtrar el lambda para cada generador
        for anden_obj in mundo_andenes.values():
            lambda_data_anden = df_lambda[
                (df_lambda_ajustado['ESTACION'] == anden_obj.estacion_nombre) &
                (df_lambda_ajustado['DIRECCION'] == anden_obj.direccion)
            ].set_index('DESDE')
            
            env.process(generador_pasajeros(env, anden_obj, lambda_data_anden, INTERVALO_LAMBDA_SEG))
            
        env.process(generador_trenes(
            env, "Hacia_Congreso", LISTA_ESTACIONES_NORTE, min_freq, max_freq,
            tiempo_viaje, dwell_time, capacidad, mundo_andenes,
            mapas_descenso_ajustados["Hacia Congreso"]
        ))
        env.process(generador_trenes(
            env, "Hacia_Catedral", LISTA_ESTACIONES_SUR, min_freq, max_freq,
            tiempo_viaje, dwell_time, capacidad, mundo_andenes,
            mapas_descenso_ajustados["Hacia Catedral"]
        ))
        
        # (NUEVO) Correr la simulación por el tiempo TOTAL (Calentamiento + Medición)
        env.run(until=TIEMPO_TOTAL_SIM)

        # Recopilación (Idéntica, porque la clase Anden ya filtró por tiempo)
        total_pax_embarcados_run = 0
        total_pax_espera_larga_run = 0
        
        for anden_obj in mundo_andenes.values():
            t_espera = anden_obj.metricas_tiempo_espera # Ya están filtrados
            l_cola = anden_obj.metricas_cola_por_tren # Ya están filtrados
            
            total_pax_embarcados_run += len(t_espera)
            esperas_largas = [t for t in t_espera if t > 300]
            total_pax_espera_larga_run += len(esperas_largas)
            
            lista_resultados_andenes.append({
                'Anden': anden_obj.nombre,
                'Replica': semilla_actual,
                'Tiempo_Espera_Promedio_Seg': np.mean(t_espera) if len(t_espera) > 0 else 0,
                'Cola_Maxima_VISTA': np.max(l_cola) if len(l_cola) > 0 else 0
            })
        
        lista_kpis_globales.append({
            'Replica': semilla_actual,
            'Total_Pasajeros_Embarcados': total_pax_embarcados_run,
            'Pasajeros_Espera_Larga (>5min)': total_pax_espera_larga_run
        })

    print("Montecarlo terminado. Analizando resultados...")
    progress_bar.empty()
    
    # --- Análisis Estadístico (Idéntico a v5) ---
    df_global_stats = pd.DataFrame(lista_resultados_andenes)
    ci_lower = lambda x: confidence_interval(x)[0]
    ci_upper = lambda x: confidence_interval(x)[1]
    df_resumen_estadistico = df_global_stats.groupby('Anden').agg(
        Cola_Max_Promedio=pd.NamedAgg(column='Cola_Maxima_VISTA', aggfunc='mean'),
        Cola_Max_CI_95_Bajo=pd.NamedAgg(column='Cola_Maxima_VISTA', aggfunc=ci_lower),
        Cola_Max_CI_95_Alto=pd.NamedAgg(column='Cola_Maxima_VISTA', aggfunc=ci_upper),
        Cola_Max_Min_Observado=pd.NamedAgg(column='Cola_Maxima_VISTA', aggfunc='min'),
        Cola_Max_Max_Observado=pd.NamedAgg(column='Cola_Maxima_VISTA', aggfunc='max'),
        Espera_Prom_Promedio_Seg=pd.NamedAgg(column='Tiempo_Espera_Promedio_Seg', aggfunc='mean'),
        Espera_Prom_CI_95_Bajo=pd.NamedAgg(column='Tiempo_Espera_Promedio_Seg', aggfunc=ci_lower),
        Espera_Prom_CI_95_Alto=pd.NamedAgg(column='Tiempo_Espera_Promedio_Seg', aggfunc=ci_upper),
        Espera_Prom_Max_Observado_Seg=pd.NamedAgg(column='Tiempo_Espera_Promedio_Seg', aggfunc='max')
    ).sort_values(by='Cola_Max_Promedio', ascending=False)
    df_resumen_estadistico = df_resumen_estadistico.round(2)
    
    df_kpis_globales = pd.DataFrame(lista_kpis_globales)
    kpi_avg_pax = df_kpis_globales['Total_Pasajeros_Embarcados'].mean()
    kpi_avg_long_wait = df_kpis_globales['Pasajeros_Espera_Larga (>5min)'].mean()
    kpi_service_level = (kpi_avg_pax - kpi_avg_long_wait) / kpi_avg_pax * 100 if kpi_avg_pax > 0 else 100
    kpi_max_long_wait_run = df_kpis_globales['Pasajeros_Espera_Larga (>5min)'].max()
    kpis_resumen = {
        'avg_pax': kpi_avg_pax,
        'avg_long_wait': kpi_avg_long_wait,
        'service_level_percent': kpi_service_level,
        'max_long_wait_run': kpi_max_long_wait_run
    }
    
    return df_resumen_estadistico, df_global_stats, kpis_resumen, df_kpis_globales


# --- 4. INTERFAZ DE USUARIO (STREAMLIT) ---
st.title("🚇 Simulador de la Línea D (Modelo TPI)")
st.write("Esta app ejecuta un modelo de Simulación de Eventos Discretos (SimPy) para la Línea D, basado en el TPI.")

# --- Barra Lateral (Idéntica a v5) ---
st.sidebar.header("Parámetros de Simulación 🛠️")
with st.sidebar.expander("1. Configuración General", expanded=True):
    n_replicaciones = st.sidebar.slider(
        "N° de Corridas (Montecarlo)", min_value=1, max_value=100, value=50, step=1
    )
with st.sidebar.expander("2. Parámetros de Oferta (Tren)", expanded=True):
    capacidad = st.sidebar.slider(
        "Capacidad del Tren (Pasajeros)", min_value=1000, max_value=2000, value=1500, step=50
    )
    dwell_time = st.sidebar.slider(
        "Tiempo de Detención (Segundos)", min_value=15, max_value=60, value=30, step=1
    )
    tiempo_viaje = st.sidebar.slider(
        "Tiempo de Viaje (Segundos)", min_value=60, max_value=180, value=120, step=5
    )
with st.sidebar.expander("3. Parámetros de Frecuencia (Tren)", expanded=False):
    min_freq = st.sidebar.slider(
        "Frecuencia Mínima (Segundos)", min_value=60, max_value=300, value=120, step=10
    )
    max_freq = st.sidebar.slider(
        "Frecuencia Mxima (Segundos)", min_value=120, max_value=360, value=180, step=10
    )
with st.sidebar.expander("4. Parámetros de Demanda (Descenso)", expanded=True):
    st.info("Ajusta las tasas de descenso base (V2/V4). 1.0 = 100% (sin cambios), 1.2 = 120% (20% más).")
    multiplicador_catedral = st.sidebar.slider(
        "Multiplicador Descenso (H. Catedral)", 
        min_value=0.5, max_value=2.0, value=1.0, step=0.05
    )
    multiplicador_congreso = st.sidebar.slider(
        "Multiplicador Descenso (H. Congreso)", 
        min_value=0.5, max_value=2.0, value=1.0, step=0.05
    )
with st.sidebar.expander("5. Multiplicador de Demanda (Arribos)", expanded=True):
    st.warning("⚠️ CRÍTICO: Multiplica la tasa de llegada de pasajeros (λ). Simula aumentos de demanda.")
    st.write("**Escenarios sugeridos:**")
    st.write("- 1.0 = Demanda actual (Base)")
    st.write("- 1.5 = Evento especial (+50%)")
    st.write("- 2.5 = Mes de alta demanda (+150%)")
    st.write("- 5.0 = Escenario extremo de estrés")
    
    multiplicador_demanda = st.sidebar.slider(
        "Multiplicador de Demanda (λ)", 
        min_value=1.0, max_value=5.0, value=1.0, step=0.1,
        help="Multiplica la cantidad de pasajeros que llegan por hora. 2.0 = el doble de demanda."
    )
    
    st.info(f"📊 Con multiplicador {multiplicador_demanda:.1f}x, se espera ~{multiplicador_demanda * 8000:.0f} pax/hora (aprox)")


# --- Lógica Principal de la App ---
st.header("1. Cargar Datos y Ejecutar Simulación")
# (NUEVO) Informar sobre el calentamiento
st.write(f"El modelo está configurado para descargar los datos y correrá con un **período de calentamiento de {TIEMPO_CALENTAMIENTO_SEG} segundos** (1 hora) ANTES de empezar a medir.")

df_lambda_final = load_and_clean_data(DATA_URL)

if df_lambda_final is not None:
    st.success("Datos de Lambda (λ) cargados y procesados exitosamente.")
    with st.expander("Ver tabla de Demanda (Lambda)"):
        st.dataframe(df_lambda_final)

    if st.button("▶️ Correr Simulación (Escenario Montecarlo)"):
        with st.spinner(f"Ejecutando {n_replicaciones} corridas (Total: {TIEMPO_TOTAL_SIM/3600*n_replicaciones:.1f} horas simuladas)..."):
            df_resumen, df_global, kpis_resumen, df_kpis_globales = run_montecarlo_simulation(
                n_replicaciones, min_freq, max_freq, capacidad, 
                dwell_time, tiempo_viaje, df_lambda_final, MAPAS_DESCENSO_BASE,
                multiplicador_catedral, multiplicador_congreso, multiplicador_demanda
            )
        st.success("¡Simulación completada!")
        st.session_state['df_resumen'] = df_resumen
        st.session_state['df_global'] = df_global
        st.session_state['kpis_resumen'] = kpis_resumen
        st.session_state['df_kpis_globales'] = df_kpis_globales
        st.session_state['n_replicaciones'] = n_replicaciones

    # --- Mostrar resultados (siempre que existan en el estado) ---
    if 'df_resumen' in st.session_state:
        df_resumen = st.session_state['df_resumen']
        df_global = st.session_state['df_global']
        kpis_resumen = st.session_state['kpis_resumen']
        df_kpis_globales = st.session_state['df_kpis_globales']
        n_replicaciones = st.session_state['n_replicaciones']

        st.header("2. Resultados del Experimento")
        st.write(f"Resultados basados en **{n_replicaciones} corridas**. Las métricas se recolectaron tras un período de calentamiento de {TIEMPO_CALENTAMIENTO_SEG}s.")

        # (NUEVO) Mostrar el escenario de demanda
        if multiplicador_demanda == 1.0:
            st.info("📊 **Escenario: Demanda Base** (sin multiplicador)")
        elif multiplicador_demanda <= 1.5:
            st.warning(f"📊 **Escenario: Evento Especial** (Demanda x{multiplicador_demanda:.1f})")
        elif multiplicador_demanda <= 2.5:
            st.warning(f"📊 **Escenario: Mes de Alta Demanda** (Demanda x{multiplicador_demanda:.1f})")
        else:
            st.error(f"⚠️ **Escenario: Estrés Extremo** (Demanda x{multiplicador_demanda:.1f})")

        st.subheader("Indicadores Clave de Rendimiento (KPIs) - Promedio por Hora")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Nivel de Servicio (Espera < 5 min)", 
            f"{kpis_resumen['service_level_percent']:.2f} %"
        )
        col2.metric(
            "Pasajeros Embarcados (Promedio por Hora)", 
            f"{kpis_resumen['avg_pax']:.0f}"
        )
        col3.metric(
            "Pasajeros con Espera Larga (>5min) (Promedio por Hora)", 
            f"{kpis_resumen['avg_long_wait']:.1f}"
        )
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen por Andén (CI 95%)", 
            "📈 Gráficos de Cuellos de Botella", 
            "🌍 Distribución (Histograma)",
            "📋 Datos Crudos"
        ])
        
        with tab1:
            st.dataframe(df_resumen)
            
        with tab2:
            st.subheader("Visualización de Cuellos de Botella (Top 10 / Bottom 10)")
            chart_mode = st.radio(
                "Seleccionar Rango de Andenes:",
                ["Top 10 (Peores Andenes)", "Bottom 10 (Mejores Andenes)"],
                horizontal=True
            )
            
            if chart_mode == "Top 10 (Peores Andenes)":
                df_chart_colas = df_resumen.sort_values(by='Cola_Max_Promedio', ascending=False).head(10)
                df_chart_espera = df_resumen.sort_values(by='Espera_Prom_Promedio_Seg', ascending=False).head(10)
                title_suffix = "(Top 10 Peores)"
            else:
                df_bottom_colas = df_resumen[df_resumen['Cola_Max_Promedio'] > 0]
                df_chart_colas = df_bottom_colas.sort_values(by='Cola_Max_Promedio', ascending=True).head(10)
                df_bottom_espera = df_resumen[df_resumen['Espera_Prom_Promedio_Seg'] > 0]
                df_chart_espera = df_bottom_espera.sort_values(by='Espera_Prom_Promedio_Seg', ascending=True).head(10)
                title_suffix = "(Bottom 10 Mejores)"

            st.write(f"**Gráfico 1: Rango de 'Cola Máxima' {title_suffix}**")
            andenes_colas = df_chart_colas.index
            data_colas = {
                'Mínimo Observado': df_chart_colas['Cola_Max_Min_Observado'],
                'Promedio': df_chart_colas['Cola_Max_Promedio'],
                'Máximo Observado': df_chart_colas['Cola_Max_Max_Observado'],
            }
            fig_cola, ax_cola = plt.subplots(figsize=(12, 7))
            x = np.arange(len(andenes_colas))
            width = 0.25
            ax_cola.bar(x - width, data_colas['Mínimo Observado'], width, label='Mínimo (Mejor Caso)', color='green')
            ax_cola.bar(x, data_colas['Promedio'], width, label='Promedio', color='orange')
            ax_cola.bar(x + width, data_colas['Máximo Observado'], width, label='Máximo (Peor Caso)', color='red')
            ax_cola.set_ylabel('N° de Pasajeros')
            ax_cola.set_title(f'Rango de "Cola Máxima" {title_suffix}')
            ax_cola.set_xticks(x)
            ax_cola.set_xticklabels(andenes_colas, rotation=90)
            ax_cola.legend()
            ax_cola.grid(axis='y', linestyle='--', alpha=0.7)
            fig_cola.tight_layout()
            st.pyplot(fig_cola)

            st.write(f"**Gráfico 2: Rango de 'Espera Promedio' {title_suffix}**")
            andenes_espera = df_chart_espera.index
            data_espera = {
                'Promedio': df_chart_espera['Espera_Prom_Promedio_Seg'],
                'Máximo Observado (Peor Promedio)': df_chart_espera['Espera_Prom_Max_Observado_Seg'],
            }
            fig_espera, ax_espera = plt.subplots(figsize=(12, 7))
            x = np.arange(len(andenes_espera))
            width = 0.35
            ax_espera.bar(x - width/2, data_espera['Promedio'], width, label='Promedio (de todos los promedios)', color='blue')
            ax_espera.bar(x + width/2, data_espera['Máximo Observado (Peor Promedio)'], width, label='Promedio Máx. Observado (Peor Corrida)', color='purple')
            ax_espera.set_ylabel('Segundos')
            ax_espera.set_title(f'Rango de "Espera Promedio" {title_suffix}')
            ax_espera.set_xticks(x)
            ax_espera.set_xticklabels(andenes_espera, rotation=90)
            ax_espera.legend()
            ax_espera.grid(axis='y', linestyle='--', alpha=0.7)
            fig_espera.tight_layout()
            st.pyplot(fig_espera)

        with tab3:
            st.subheader("Análisis Específico: Distribución de Resultados")
            st.write(f"Analizá la variabilidad de las {n_replicaciones} corridas para un andén específico.")
            anden_seleccionado = st.selectbox(
                "Seleccioná un Andén para analizar:",
                df_resumen.index.sort_values()
            )
            if anden_seleccionado:
                data_anden = df_global[df_global['Anden'] == anden_seleccionado]
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Distribución de 'Cola Máxima' (Histograma)**")
                    fig_cola_hist, ax_cola_hist = plt.subplots()
                    ax_cola_hist.hist(data_anden['Cola_Maxima_VISTA'], bins=15, alpha=0.7, edgecolor='black')
                    ax_cola_hist.set_title(f"Distribución 'Cola Máxima' en {anden_seleccionado}")
                    ax_cola_hist.set_xlabel("Cola Máxima (Pasajeros)")
                    ax_cola_hist.set_ylabel(f"Frecuencia (de {n_replicaciones} corridas)")
                    st.pyplot(fig_cola_hist)
                with col2:
                    st.write("**Distribución de 'Espera Promedio' (Histograma)**")
                    fig_espera_hist, ax_espera_hist = plt.subplots()
                    ax_espera_hist.hist(data_anden['Tiempo_Espera_Promedio_Seg'], bins=15, alpha=0.7, color='blue', edgecolor='black')
                    ax_espera_hist.set_title(f"Distribución 'Espera Promedio' en {anden_seleccionado}")
                    ax_espera_hist.set_xlabel("Espera Promedio (Segundos)")
                    ax_espera_hist.set_ylabel(f"Frecuencia (de {n_replicaciones} corridas)")
                    st.pyplot(fig_espera_hist)
        
        with tab4:
            st.subheader("Datos Crudos de las Réplicas")
            st.write(f"Resultados detallados de los KPIs globales para todas las {n_replicaciones} corridas.")
            st.dataframe(df_kpis_globales)
        
        st.info("Simulación finalizada. Podés cambiar los parámetros de la barra lateral y volver a correr.")
else:
    st.error("No se pudieron cargar los datos de Lambda. Revisa el link en DATA_URL.")
