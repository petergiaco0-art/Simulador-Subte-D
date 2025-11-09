import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats # (Importante, nos aseguramos que esté scipy)

# --- 0. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Simulador Subte Línea D",
    layout="wide"
)

# --- URL de Datos (Reemplazar con tu link de Release) ---
DATA_URL = "https://github.com/petergiaco0-art/Simulador-Subte-D/releases/download/v1.0/202501_D.csv" 

# --- 1. FUNCIONES DE CARGA Y DATOS ---
@st.cache_data
def load_and_clean_data(url_de_datos):
    st.info(f"Descargando y procesando datos de demanda (Enero)...")
    print("Iniciando descarga y limpieza de datos...")
    
    # ... (El código de carga, limpieza y cálculo de Lambda es IDÉNTICO a la v4) ...
    # --- Carga de Datos ---
    try:
        df = pd.read_csv(
            url_de_datos, sep=';', encoding='utf-8-sig',
            quoting=csv.QUOTE_NONE, low_memory=False
        )
    except Exception as e:
        st.error(f"Error fatal al leer el CSV desde la URL: {e}")
        return None
    # --- Limpieza (Idéntica a antes) ---
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
    print("Carga y limpieza de datos completada.")
    # --- Cálculo de Lambda (Idéntico a antes) ---
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
# (CONSTANTES, Clase Anden, Procesos: idénticos a v4)
LISTA_ESTACIONES_NORTE = [
    'Catedral', '9 de Julio', 'Tribunales', 'Callao', 'Facultad de Medicina', 
    'Agüero', 'Pueyrredon.D', 'Bulnes', 'Plaza Italia', 'Ministro Carranza', 
    'Olleros', 'Jose Hernandez', 'Juramento', 'Congreso de Tucuman'
]
LISTA_ESTACIONES_SUR = LISTA_ESTACIONES_NORTE[::-1]
TIEMPO_SIMULACION_SEG = 3600 # 1 Hora
INTERVALO_LAMBDA_SEG = 900 # 15 min
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
class Anden:
    def __init__(self, env, nombre_estacion, direccion):
        self.env = env
        self.nombre = f"{nombre_estacion}_{direccion}"
        self.direccion = direccion
        self.estacion_nombre = nombre_estacion
        self.cola_pasajeros = simpy.Store(env)
        self.metricas_tiempo_espera = []
        self.metricas_cola_por_tren = []
    def add_pasajero(self, pasajero):
        pasajero['tiempo_llegada_anden'] = self.env.now
        return self.cola_pasajeros.put(pasajero)
    def tren_llega_a_anden(self, tren, dwell_time):
        pasajeros_en_cola_al_llegar = len(self.cola_pasajeros.items)
        self.metricas_cola_por_tren.append(pasajeros_en_cola_al_llegar)
        pasajeros_subidos = 0
        while tren['pasajeros_actuales'] < tren['capacidad'] and len(self.cola_pasajeros.items) > 0:
            pasajero = yield self.cola_pasajeros.get()
            tiempo_espera = self.env.now - pasajero['tiempo_llegada_anden']
            self.metricas_tiempo_espera.append(tiempo_espera) # Guardamos CADA espera
            tren['pasajeros_actuales'] += 1
            pasajeros_subidos += 1
        yield self.env.timeout(dwell_time)
def generador_pasajeros(env, anden, df_lambda, intervalo_seg):
    lambda_data = df_lambda[
        (df_lambda['ESTACION'] == anden.estacion_nombre) &
        (df_lambda['DIRECCION'] == anden.direccion)
    ].set_index('DESDE') 
    intervalos = ['08:00:00', '08:15:00', '08:30:00', '08:45:00']
    for intervalo_hora in intervalos:
        lambda_actual = 0.0
        try: lambda_actual = lambda_data.loc[intervalo_hora]['pasajeros_PROMEDIO_lambda']
        except KeyError: pass
        if lambda_actual > 0:
            seg_entre_pasajeros = intervalo_seg / lambda_actual
            inicio_intervalo = env.now
            while env.now < inicio_intervalo + intervalo_seg:
                tiempo_llegada_prox_pax = random.expovariate(1.0 / seg_entre_pasajeros)
                yield env.timeout(tiempo_llegada_prox_pax)
                if env.now < inicio_intervalo + intervalo_seg:
                    pasajero = {'id': f"pax_{random.randint(1000,9999)}"}
                    yield anden.add_pasajero(pasajero)
        else:
            yield env.timeout(intervalo_seg)
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
def generador_trenes(env, direccion, lista_recorrido, min_freq, max_freq, tiempo_viaje, dwell_time, capacidad, mundo_andenes, tasas_descenso):
    tren_id_counter = 0
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

# --- (MODIFICACIÓN: Función Montecarlo ahora calcula KPIs) ---
def run_montecarlo_simulation(
    n_replicaciones, min_freq, max_freq, capacidad, dwell_time, tiempo_viaje, 
    df_lambda, mapas_descenso_base, 
    multiplicador_catedral, multiplicador_congreso
    ):
    
    print(f"Iniciando Montecarlo con {n_replicaciones} réplicas...")
    lista_resultados_andenes = []
    lista_kpis_globales = [] # <-- (NUEVO) Lista para KPIs
    
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
    
    progress_bar = st.progress(0, text="Iniciando simulación...")

    # Bucle de Replicación (Idéntico)
    for i in range(n_replicaciones):
        semilla_actual = i + 1
        progress_bar.progress((i + 1) / n_replicaciones, text=f"Corriendo Réplica {semilla_actual}/{n_replicaciones}...")
        
        random.seed(semilla_actual)
        np.random.seed(semilla_actual)
        
        env = simpy.Environment()
        mundo_andenes = {}
        for estacion in LISTA_ESTACIONES_NORTE:
            anden_norte = Anden(env, estacion, "Hacia Congreso") 
            mundo_andenes[anden_norte.nombre] = anden_norte
            anden_sur = Anden(env, estacion, "Hacia Catedral") 
            mundo_andenes[anden_sur.nombre] = anden_sur
            
        for anden_obj in mundo_andenes.values():
            env.process(generador_pasajeros(env, anden_obj, df_lambda, INTERVALO_LAMBDA_SEG))
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
        
        env.run(until=TIEMPO_SIMULACION_SEG)

        # --- (NUEVO) Recopilación de KPIs para ESTA corrida ---
        total_pax_embarcados_run = 0
        total_pax_espera_larga_run = 0
        
        # Recopilar resultados de esta réplica (Andenes)
        for anden_obj in mundo_andenes.values():
            t_espera = anden_obj.metricas_tiempo_espera
            l_cola = anden_obj.metricas_cola_por_tren
            
            # Sumar al KPI
            total_pax_embarcados_run += len(t_espera)
            # Contar cuántos en esta lista son > 300 segundos
            esperas_largas = [t for t in t_espera if t > 300]
            total_pax_espera_larga_run += len(esperas_largas)
            
            # Guardar datos por andén
            lista_resultados_andenes.append({
                'Anden': anden_obj.nombre,
                'Replica': semilla_actual,
                'Tiempo_Espera_Promedio_Seg': np.mean(t_espera) if len(t_espera) > 0 else 0,
                'Cola_Maxima_VISTA': np.max(l_cola) if len(l_cola) > 0 else 0
            })
        
        # Guardar KPIs de esta corrida
        lista_kpis_globales.append({
            'Replica': semilla_actual,
            'Total_Pasajeros_Embarcados': total_pax_embarcados_run,
            'Pasajeros_Espera_Larga (>5min)': total_pax_espera_larga_run
        })
        # --- FIN DE SECCIÓN NUEVA ---

    print("Montecarlo terminado. Analizando resultados...")
    progress_bar.empty()
    
    # --- Análisis Estadístico (Andenes) ---
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
    
    # --- (NUEVO) Análisis Estadístico (KPIs Globales) ---
    df_kpis_globales = pd.DataFrame(lista_kpis_globales)
    
    # Calcular promedios (lo que pediste)
    kpi_avg_pax = df_kpis_globales['Total_Pasajeros_Embarcados'].mean()
    kpi_avg_long_wait = df_kpis_globales['Pasajeros_Espera_Larga (>5min)'].mean()
    
    # Calcular Nivel de Servicio
    if kpi_avg_pax > 0:
        kpi_service_level = (kpi_avg_pax - kpi_avg_long_wait) / kpi_avg_pax * 100
    else:
        kpi_service_level = 100
    
    # Peor caso observado en las 50 corridas
    kpi_max_long_wait_run = df_kpis_globales['Pasajeros_Espera_Larga (>5min)'].max()

    kpis_resumen = {
        'avg_pax': kpi_avg_pax,
        'avg_long_wait': kpi_avg_long_wait,
        'service_level_percent': kpi_service_level,
        'max_long_wait_run': kpi_max_long_wait_run
    }
    
    # Devolvemos todo
    return df_resumen_estadistico, df_global_stats, kpis_resumen, df_kpis_globales


# --- 4. INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🚇 Simulador de la Línea D (Modelo TPI)")
st.write("Esta app ejecuta un modelo de Simulación de Eventos Discretos (SimPy) para la Línea D, basado en el TPI.")

# --- Barra Lateral (Idéntica a v4) ---
st.sidebar.header("Parámetros de Simulación 🛠️")
with st.sidebar.expander("1. Configuración General", expanded=True):
    n_replicaciones = st.sidebar.slider(
        "N° de Corridas (Montecarlo)", min_value=1, max_value=100, value=50, step=1,
        help="Número de veces que se corre la simulación (con diferentes semillas) para obtener un resultado estadístico."
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
        "Frecuencia Máxima (Segundos)", min_value=120, max_value=360, value=180, step=10
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

# --- Lógica Principal de la App ---
st.header("1. Cargar Datos y Ejecutar Simulación")
st.write("El modelo está configurado para descargar los datos de demanda (Enero) desde GitHub.")

# Cargar y procesar datos automáticamente
df_lambda_final = load_and_clean_data(DATA_URL)

if df_lambda_final is not None:
    st.success("Datos de Lambda (λ) cargados y procesados exitosamente.")
    with st.expander("Ver tabla de Demanda (Lambda)"):
        st.dataframe(df_lambda_final)

    # Botón para correr la simulación
    if st.button("▶️ Correr Simulación (Escenario Montecarlo)"):
        
        with st.spinner(f"Ejecutando {n_replicaciones} corridas..."):
            # --- (MODIFICADO) Capturar los 4 outputs ---
            df_resumen, df_global, kpis_resumen, df_kpis_globales = run_montecarlo_simulation(
                n_replicaciones, min_freq, max_freq, capacidad, 
                dwell_time, tiempo_viaje, df_lambda_final, MAPAS_DESCENSO_BASE,
                multiplicador_catedral, multiplicador_congreso
            )
        
        st.success("¡Simulación completada!")
        st.session_state['df_resumen'] = df_resumen
        st.session_state['df_global'] = df_global
        st.session_state['kpis_resumen'] = kpis_resumen
        st.session_state['df_kpis_globales'] = df_kpis_globales
        st.session_state['n_replicaciones'] = n_replicaciones

    # --- (MODIFICACIÓN: Interfaz de Resultados con KPIs) ---
    if 'df_resumen' in st.session_state:
        df_resumen = st.session_state['df_resumen']
        df_global = st.session_state['df_global']
        kpis_resumen = st.session_state['kpis_resumen']
        df_kpis_globales = st.session_state['df_kpis_globales']
        n_replicaciones = st.session_state['n_replicaciones']

        st.header("2. Resultados del Experimento")
        st.write(f"Resultados basados en **{n_replicaciones} corridas** (réplicas) de la simulación de 1 hora.")

        # --- (NUEVO) Sección de KPIs Globales ---
        st.subheader("Indicadores Clave de Rendimiento (KPIs) - Promedio por Hora")
        st.write("Estos son los promedios de todo el sistema, calculados a partir de las 50 corridas.")
        
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
        # --- Fin Sección KPIs ---

        # Pestañas de Resultados
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen por Andén (CI 95%)", 
            "📈 Gráficos de Cuellos de Botella", 
            "🌍 Distribución (Histograma)",
            "📋 Datos Crudos"
        ])

        with tab1:
            st.subheader("Resumen Estadístico por Andén")
            st.write(f"Resultados basados en {n_replicaciones} corridas, mostrando la media e Intervalos de Confianza del 95%.")
            st.dataframe(df_resumen)
            with st.expander("Detalle de las columnas"):
                st.markdown(f"""
                - **Cola_Max_Promedio**: La media de las colas máximas observadas en las {n_replicaciones} corridas.
                - **Cola_Max_CI_95_Bajo / Alto**: **Intervalo de Confianza del 95%**. Tenemos 95% de confianza de que la *verdadera* media de la cola máxima del sistema está entre estos two valores.
                - **Cola_Max_Min_Observado**: La cola máxima más *baja* que se vio en la "mejor" corrida.
                - **Cola_Max_Max_Observado**: La cola máxima más *alta* que se vio en la "peor" corrida.
                - **Espera_Prom_...**: Mismas métricas, pero para el tiempo de espera promedio.
                """)

        with tab2:
            st.subheader("Visualización de Cuellos de Botella")
            st.write("Gráficos de barras agrupadas mostrando el rango de resultados (Mínimo, Promedio y Máximo) de las 50 corridas.")
            
            # --- (NUEVO) Selector Top/Bottom ---
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
                # Excluimos los andenes con 0 (terminales) para que el gráfico sea útil
                df_bottom_colas = df_resumen[df_resumen['Cola_Max_Promedio'] > 0]
                df_chart_colas = df_bottom_colas.sort_values(by='Cola_Max_Promedio', ascending=True).head(10)
                
                df_bottom_espera = df_resumen[df_resumen['Espera_Prom_Promedio_Seg'] > 0]
                df_chart_espera = df_bottom_espera.sort_values(by='Espera_Prom_Promedio_Seg', ascending=True).head(10)
                title_suffix = "(Bottom 10 Mejores)"

            # --- Gráfico 1: Colas (Agrupado) ---
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
            rects1 = ax_cola.bar(x - width, data_colas['Mínimo Observado'], width, label='Mínimo (Mejor Caso)', color='green')
            rects2 = ax_cola.bar(x, data_colas['Promedio'], width, label='Promedio', color='orange')
            rects3 = ax_cola.bar(x + width, data_colas['Máximo Observado'], width, label='Máximo (Peor Caso)', color='red')
            ax_cola.set_ylabel('N° de Pasajeros')
            ax_cola.set_title(f'Rango de "Cola Máxima" {title_suffix}')
            ax_cola.set_xticks(x)
            ax_cola.set_xticklabels(andenes_colas, rotation=90)
            ax_cola.legend()
            ax_cola.grid(axis='y', linestyle='--', alpha=0.7)
            fig_cola.tight_layout()
            st.pyplot(fig_cola)

            # --- Gráfico 2: Esperas (Agrupado) ---
            st.write(f"**Gráfico 2: Rango de 'Espera Promedio' {title_suffix}**")
            andenes_espera = df_chart_espera.index
            data_espera = {
                'Promedio': df_chart_espera['Espera_Prom_Promedio_Seg'],
                'Máximo Observado (Peor Promedio)': df_chart_espera['Espera_Prom_Max_Observado_Seg'],
            }
            fig_espera, ax_espera = plt.subplots(figsize=(12, 7))
            x = np.arange(len(andenes_espera))
            width = 0.35
            rects1 = ax_espera.bar(x - width/2, data_espera['Promedio'], width, label='Promedio (de todos los promedios)', color='blue')
            rects2 = ax_espera.bar(x + width/2, data_espera['Máximo Observado (Peor Promedio)'], width, label='Promedio Máx. Observado (Peor Corrida)', color='purple')
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
            st.write(f"Resultados detallados de todas las {n_replicaciones} corridas (Total {len(df_global)} filas).")
            # Mostramos el df_kpis_globales, que es el resumen por corrida
            st.dataframe(df_kpis_globales)
            
        st.info("Simulación finalizada. Podés cambiar los parámetros de la barra lateral y volver a correr.")
else:
    st.error("No se pudieron cargar los datos de Lambda. Revisa el link en DATA_URL.")
