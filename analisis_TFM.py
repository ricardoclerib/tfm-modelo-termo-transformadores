"""
Analisis de datos operativos de subestaciones
TFM Cleri Bravo

Identifica los parametros del modelo termico de la norma IEEE C57.91 a partir
de los registros diarios de operacion de dos subestaciones, y comprueba si
ese modelo describe el comportamiento real de los transformadores.

El programa corre de principio a fin sobre los conjuntos ya anonimizados de
la carpeta datos/ y produce:

    resultados.json     ajuste, validacion y envejecimiento de cada maquina
    sintetico.json      el experimento de identificabilidad
    condicion.json      los demas indicadores de condicion de los registros
    figuras/            las figuras del trabajo, en PDF y en PNG

Uso:

    python analisis_TFM.py [carpeta_datos] [carpeta_salida]

Sin argumentos, busca la carpeta datos/ junto al programa y escribe en
salida/, tambien junto al programa.
"""

import csv
import json
import math
import os
import re
import sys
import datetime as dt
from collections import defaultdict

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as falta:
    sys.exit("Falta una libreria: %s\n"
             "Instalar con:  pip install numpy matplotlib" % falta.name)

# ---------------------------------------------------------------- constantes

# Valores normativos para refrigeracion con aire forzado. Se mantienen fijos.
R_NOMINAL = 3.2
N_NOMINAL = 0.8

# Rejilla de busqueda de la potencia nominal, en por unidad de la placa.
REJILLA_SR = np.linspace(0.30, 3.00, 271)
TOLERANCIA_EQUIVALENCIA = 0.02

# Dias con al menos esta cobertura horaria se consideran de cobertura completa.
COBERTURA_COMPLETA = 18

MAQUINAS = ("T1", "T2", "T3")
VARIANTES = {"maximo": "aceite_max_c", "ronda": "aceite_ronda_c"}
BOBINADO = {"maximo": "bobinado_max_c", "ronda": "bobinado_ronda_c"}

COLOR = {"T1": "#1f6fb4", "T2": "#c2600a", "T3": "#8f3f8f"}
MARCA = {"T1": "o", "T2": "s", "T3": "^"}
TRAZO = {"T1": "-", "T2": "--", "T3": "-."}
TINTA = "#2b2b2b"
TINTA_SUAVE = "#6b6b6b"


# ------------------------------------------------------------------- lectura

def leer_conjunto(ruta):
    """Lee un archivo de datos y convierte a numero lo que sea numero."""
    filas = []
    with open(ruta, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            limpia = {}
            for clave, valor in fila.items():
                if valor is None or valor == "":
                    limpia[clave] = None
                elif clave in ("transformador", "subestacion", "fecha"):
                    limpia[clave] = valor
                else:
                    try:
                        limpia[clave] = float(valor)
                    except ValueError:
                        limpia[clave] = None
            filas.append(limpia)
    return filas


def vectores(filas, campo_aceite):
    """Extrae las series que necesita el modelo, descartando dias incompletos."""
    utiles = [f for f in filas
              if f.get(campo_aceite) is not None
              and f.get("ambiente_c") is not None
              and f.get("carga_pu") is not None]
    carga = np.array([f["carga_pu"] for f in utiles])
    aceite = np.array([f[campo_aceite] for f in utiles])
    ambiente = np.array([f["ambiente_c"] for f in utiles])
    meses = [f["fecha"][:7] for f in utiles]
    periodos = np.array([int(f["periodo"]) for f in utiles])
    return utiles, carga, aceite, ambiente, meses, periodos


# -------------------------------------------------------------------- modelo

def factor_carga(carga_pu, sr_pu, R=R_NOMINAL, n=N_NOMINAL):
    """Termino adimensional del modelo normativo."""
    K = np.asarray(carga_pu, dtype=float) / float(sr_pu)
    return ((K ** 2 * R + 1.0) / (R + 1.0)) ** n


def delta_optimo(elevacion, f):
    """delta_R por minimos cuadrados, con la potencia nominal ya fijada."""
    denominador = float(np.sum(f * f))
    if denominador <= 0.0:
        return 0.0
    return float(np.sum(elevacion * f) / denominador)


def ajustar_con_placa(carga, aceite, ambiente, sr_pu=1.0):
    """Ajuste principal: potencia fijada en la placa, un solo parametro libre."""
    elevacion = np.asarray(aceite, float) - np.asarray(ambiente, float)
    f = factor_carga(carga, sr_pu)
    delta = delta_optimo(elevacion, f)
    residuo = elevacion - delta * f
    return {"sr_pu": float(sr_pu), "delta_r": delta,
            "mae": float(np.mean(np.abs(residuo))),
            "rmse": float(np.sqrt(np.mean(residuo ** 2))),
            "parametros_libres": 1}


def ajustar_libre(carga, aceite, ambiente, rejilla=REJILLA_SR):
    """Ajuste con los dos parametros libres. Solo para estudiar identificabilidad."""
    elevacion = np.asarray(aceite, float) - np.asarray(ambiente, float)
    mejor, curva = None, []
    for sr in rejilla:
        f = factor_carga(carga, sr)
        delta = delta_optimo(elevacion, f)
        mae = float(np.mean(np.abs(elevacion - delta * f)))
        curva.append((float(sr), delta, mae))
        if mejor is None or mae < mejor["mae"]:
            mejor = {"sr_pu": float(sr), "delta_r": delta, "mae": mae}
    mejor["curva"] = curva
    mejor["en_el_borde"] = bool(mejor["sr_pu"] <= rejilla[0] + 1e-9
                                or mejor["sr_pu"] >= rejilla[-1] - 1e-9)
    return mejor


def intervalo_equivalencia(ajuste, tolerancia=TOLERANCIA_EQUIVALENCIA):
    """Valores de la potencia nominal cuyo error no se distingue del minimo."""
    limite = ajuste["mae"] * (1.0 + tolerancia)
    admisibles = [sr for sr, _, mae in ajuste["curva"] if mae <= limite]
    if not admisibles:
        return (ajuste["sr_pu"], ajuste["sr_pu"])
    return (min(admisibles), max(admisibles))


def predecir(carga, ambiente, sr_pu, delta_r):
    return np.asarray(ambiente, float) + delta_r * factor_carga(carga, sr_pu)


def ajustar_lineal(carga, aceite, ambiente):
    """Contraste con dos parametros y sin justificacion fisica."""
    x = np.asarray(carga, float)
    y = np.asarray(aceite, float) - np.asarray(ambiente, float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residuo = y - A @ coef
    return {"intercepto": float(coef[0]), "pendiente": float(coef[1]),
            "mae": float(np.mean(np.abs(residuo)))}


def predecir_lineal(carga, ambiente, ajuste):
    return (np.asarray(ambiente, float) + ajuste["intercepto"]
            + ajuste["pendiente"] * np.asarray(carga, float))


def ajustar_media(aceite, ambiente):
    """Linea base: elevacion constante sobre el ambiente."""
    y = np.asarray(aceite, float) - np.asarray(ambiente, float)
    m = float(np.mean(y))
    return {"elevacion_media": m, "mae": float(np.mean(np.abs(y - m)))}


def factor_envejecimiento(theta_punto_caliente):
    """Expresion de Arrhenius de la norma, referida a 110 grados."""
    t = np.asarray(theta_punto_caliente, float)
    return np.exp(15000.0 / 383.0 - 15000.0 / (t + 273.0))


# ---------------------------------------------------------------- validacion

def validacion_cruzada(carga, aceite, ambiente, meses):
    """Error fuera de muestra dejando un mes completo fuera cada vez."""
    indices = defaultdict(list)
    for i, m in enumerate(meses):
        indices[m].append(i)

    errores = {"placa": [], "libre": [], "lineal": [], "media": []}
    for mes in sorted(indices):
        fuera = np.array(indices[mes])
        conjunto = set(fuera.tolist())
        dentro = np.array([i for i in range(len(carga)) if i not in conjunto])
        if len(dentro) < 10 or len(fuera) == 0:
            continue

        pl = ajustar_con_placa(carga[dentro], aceite[dentro], ambiente[dentro])
        errores["placa"] += np.abs(
            aceite[fuera] - predecir(carga[fuera], ambiente[fuera],
                                     pl["sr_pu"], pl["delta_r"])).tolist()

        li = ajustar_libre(carga[dentro], aceite[dentro], ambiente[dentro])
        errores["libre"] += np.abs(
            aceite[fuera] - predecir(carga[fuera], ambiente[fuera],
                                     li["sr_pu"], li["delta_r"])).tolist()

        ln = ajustar_lineal(carga[dentro], aceite[dentro], ambiente[dentro])
        errores["lineal"] += np.abs(
            aceite[fuera] - predecir_lineal(carga[fuera], ambiente[fuera],
                                            ln)).tolist()

        me = ajustar_media(aceite[dentro], ambiente[dentro])
        errores["media"] += np.abs(
            aceite[fuera] - (ambiente[fuera] + me["elevacion_media"])).tolist()

    return {k: (float(np.mean(v)) if v else None) for k, v in errores.items()}


def validacion_temporal(carga, aceite, ambiente, periodos):
    """Ajusta en el primer periodo y mide en el segundo."""
    a, b = periodos == 0, periodos == 1
    if a.sum() < 15 or b.sum() < 15:
        return None
    pl = ajustar_con_placa(carga[a], aceite[a], ambiente[a])
    ln = ajustar_lineal(carga[a], aceite[a], ambiente[a])
    return {
        "dias_ajuste": int(a.sum()),
        "dias_prueba": int(b.sum()),
        "delta_r_del_primer_periodo": pl["delta_r"],
        "mae_con_placa": float(np.mean(np.abs(
            aceite[b] - predecir(carga[b], ambiente[b],
                                 pl["sr_pu"], pl["delta_r"])))),
        "mae_lineal": float(np.mean(np.abs(
            aceite[b] - predecir_lineal(carga[b], ambiente[b], ln)))),
    }


# ------------------------------------------------- experimento de identificabilidad

SR_VERDADERO = 1.00
DELTA_VERDADERO = 55.0
AMBIENTE_MEDIO, AMBIENTE_DESVIACION = 27.0, 2.0
CARGA_CENTRAL = 0.55
RUIDO_ACEITE = 1.0
DIAS_SERIE = 200
REPETICIONES = 200
RECORRIDOS = [0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
ERRORES_AMBIENTE = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def serie_artificial(recorrido, error_ambiente, rng):
    """Genera una serie con parametros conocidos y error de medida controlado."""
    mitad = recorrido / 2.0
    carga = rng.uniform(CARGA_CENTRAL - mitad, CARGA_CENTRAL + mitad, DIAS_SERIE)
    ambiente_real = rng.normal(AMBIENTE_MEDIO, AMBIENTE_DESVIACION, DIAS_SERIE)
    aceite = (ambiente_real + DELTA_VERDADERO * factor_carga(carga, SR_VERDADERO)
              + rng.normal(0.0, RUIDO_ACEITE, DIAS_SERIE))
    ambiente_registrado = ambiente_real + rng.normal(0.0, error_ambiente,
                                                     DIAS_SERIE)
    return carga, np.round(aceite), np.round(ambiente_registrado)


def experimento_identificabilidad():
    """Mide el error de recuperacion de la potencia nominal al variar
    el recorrido de carga y el error de medida del ambiente."""
    celdas, semilla = [], 0
    for i, recorrido in enumerate(RECORRIDOS, start=1):
        print("   recorrido %.2f pu  (%d de %d)" % (recorrido, i, len(RECORRIDOS)),
              flush=True)
        for error in ERRORES_AMBIENTE:
            semilla += 1
            rng = np.random.default_rng(semilla)
            errores, pendientes, en_borde = [], [], 0
            for _ in range(REPETICIONES):
                carga, aceite, ambiente = serie_artificial(recorrido, error, rng)
                aj = ajustar_libre(carga, aceite, ambiente)
                errores.append(abs(aj["sr_pu"] - SR_VERDADERO)
                               / SR_VERDADERO * 100.0)
                if aj["en_el_borde"]:
                    en_borde += 1
                pl = ajustar_con_placa(carga, aceite, ambiente)
                resid = ((aceite - ambiente)
                         - pl["delta_r"] * factor_carga(carga, 1.0))
                pendientes.append(float(np.polyfit(ambiente, resid, 1)[0]))
            celdas.append({
                "recorrido": recorrido,
                "error_ambiente": error,
                "error_sr_mediana_pct": float(np.median(errores)),
                "error_sr_p90_pct": float(np.percentile(errores, 90)),
                "proporcion_en_el_borde": en_borde / REPETICIONES,
                "pendiente_residuo_mediana": float(np.median(pendientes)),
            })
    return {"sr_verdadero_pu": SR_VERDADERO,
            "delta_verdadero_c": DELTA_VERDADERO,
            "ruido_aceite_c": RUIDO_ACEITE,
            "dias_por_serie": DIAS_SERIE,
            "repeticiones": REPETICIONES,
            "celdas": celdas}


# --------------------------------------------------------- estado de condicion
#
# Indicadores que los mismos formularios contienen y que no dependen del
# modelo termico: la diferencia entre las dos lecturas del aceite, el
# gradiente entre bobinado y aceite, la deriva del residuo en el tiempo, el
# ritmo del cambiador de tomas, los gases disueltos, la coherencia del
# ambiente entre las dos subestaciones y la parte del aceite de S1 que el
# ambiente registrado no explica.

# Umbrales de maniobras del cambiador de tomas que los fabricantes suelen
# fijar para la revision del ruptor.
UMBRALES_MANIOBRAS = (300000, 500000)

GASES = ("H2", "CH4", "C2H4", "C2H6", "CO", "CO2", "C2H2")


def mediana_de_pendientes(x, y, maximo_pares=200000):
    """Estimador de Theil-Sen: mediana de las pendientes entre pares.

    El contador del cambiador contiene errores de transcripcion de un digito,
    y uno solo desplaza la recta de minimos cuadrados mas que todo el
    crecimiento real del periodo. La mediana de las pendientes tolera que una
    minoria de puntos este corrompida.
    """
    n = len(x)
    paso = 1
    if n * (n - 1) // 2 > maximo_pares:
        paso = int(math.ceil(n / math.sqrt(2.0 * maximo_pares)))
    pendientes = []
    for i in range(0, n, paso):
        for j in range(i + 1, n, paso):
            dx = x[j] - x[i]
            if dx > 0:
                pendientes.append((y[j] - y[i]) / dx)
    return float(np.median(pendientes)) if pendientes else 0.0


def maximo_frente_a_ronda(filas):
    pares = [(f["aceite_ronda_c"], f["aceite_max_c"]) for f in filas
             if f["aceite_ronda_c"] is not None
             and f["aceite_max_c"] is not None]
    if not pares:
        return None
    dif = np.array([b - a for a, b in pares])
    return {"dias": len(pares),
            "diferencia_mediana": float(np.median(dif)),
            "proporcion_maximo_mayor": float(np.mean(dif > 0))}


def gradiente_bobinado(filas):
    """Bobinado menos aceite. En regimen permanente no puede ser negativo."""
    pares = [(f["carga_pu"], f["bobinado_max_c"] - f["aceite_max_c"])
             for f in filas
             if f.get("aceite_max_c") is not None
             and f.get("bobinado_max_c") is not None
             and f.get("carga_pu") is not None]
    if len(pares) < 20:
        return None
    k = np.array([p[0] for p in pares])
    g = np.array([p[1] for p in pares])
    return {"dias": len(pares),
            "mediana": float(np.median(g)),
            "invertidos": int(np.sum(g < 0)),
            "proporcion_invertidos": float(np.mean(g < 0)),
            "minimo": float(g.min()),
            "pendiente_frente_a_k2": float(np.polyfit(k ** 2, g, 1)[0])}


def residuo_en_el_tiempo(filas):
    """Residuo del modelo con la placa fijada y el tiempo en meses."""
    u, carga, aceite, amb, _, _ = vectores(filas, "aceite_max_c")
    pl = ajustar_con_placa(carga, aceite, amb)
    resid = (aceite - amb) - pl["delta_r"] * factor_carga(carga, 1.0)
    t = np.array([dt.date.fromisoformat(f["fecha"]).toordinal() for f in u],
                 dtype=float)
    return u, (t - t.min()) / 30.44, resid


def deriva(filas):
    _, t, resid = residuo_en_el_tiempo(filas)
    coef, cov = np.polyfit(t, resid, 1, cov=True)
    error = float(np.sqrt(cov[0, 0]))
    return {"grados_por_mes": float(coef[0]),
            "error_tipico": error,
            "estadistico_t": float(coef[0]) / error,
            "cambio_en_el_periodo": float(coef[0]) * float(t.max())}


def ritmo_cambiador(filas):
    puntos = sorted((f["fecha"], f["oltc_operaciones"]) for f in filas
                    if f.get("oltc_operaciones") is not None
                    and f["oltc_operaciones"] > 0)
    if len(puntos) < 30:
        return None
    fechas = np.array([dt.date.fromisoformat(p[0]).toordinal()
                       for p in puntos], dtype=float)
    contador = np.array([p[1] for p in puntos], dtype=float)
    # El contador es acumulado: cada retroceso es un error de transcripcion.
    retrocesos = int(np.sum(np.diff(contador) < 0))
    por_dia = mediana_de_pendientes(fechas, contador)
    por_anio = por_dia * 365.25
    actual = float(np.median(contador[-5:]))
    anios = {}
    for umbral in UMBRALES_MANIOBRAS:
        anios[str(umbral)] = (round((umbral - actual) / por_anio, 1)
                              if por_anio > 0 and umbral > actual else None)
    return {"dias": len(fechas),
            "retrocesos_del_contador": retrocesos,
            "proporcion_retrocesos": retrocesos / max(len(fechas) - 1, 1),
            "contador_actual": actual,
            "maniobras_por_anio": float(por_anio),
            "anios_hasta_umbral": anios}


def regresion_con_ambiente(filas):
    """Coeficientes de aceite = a + b*carga + c*ambiente.

    La norma suma el ambiente sin ponderarlo, asi que supone c = 1. Se calcula
    con el maximo diario del aceite, que es lo que describe el modelo, y con la
    lectura de ronda, que el operador toma en el mismo momento que el ambiente.
    """
    _, carga, aceite, amb, _, _ = vectores(filas, "aceite_max_c")
    A = np.column_stack([np.ones_like(carga), carga, amb])
    coef, *_ = np.linalg.lstsq(A, aceite, rcond=None)
    _, carga_r, aceite_r, amb_r, _, _ = vectores(filas, "aceite_ronda_c")
    A_r = np.column_stack([np.ones_like(carga_r), carga_r, amb_r])
    coef_r, *_ = np.linalg.lstsq(A_r, aceite_r, rcond=None)
    return {"coeficiente_ambiente": float(coef[2]),
            "coeficiente_carga": float(coef[1]),
            "correlacion_aceite_carga": float(np.corrcoef(aceite, carga)[0, 1]),
            "dias": len(carga),
            "coeficiente_ambiente_ronda": float(coef_r[2]),
            "dias_ronda": len(carga_r)}


def gases_disueltos(filas):
    salida = {}
    for gas in GASES:
        v = [f["gas_" + gas] for f in filas if f.get("gas_" + gas) is not None]
        if v:
            salida[gas] = {"lecturas": len(v), "minimo": float(min(v)),
                           "maximo": float(max(v)),
                           "mediana": float(np.median(v))}
    # Una lectura de acetileno es un valor o una marca de limite de deteccion.
    # La proporcion se calcula sobre las lecturas, no sobre todos los dias.
    bajo = sum(1 for f in filas if f.get("gas_C2H2_ldl") == 1)
    con_valor = sum(1 for f in filas if f.get("gas_C2H2") is not None)
    if bajo + con_valor:
        salida["C2H2_bajo_el_limite"] = {
            "lecturas": bajo, "lecturas_de_acetileno": bajo + con_valor,
            "proporcion": bajo / (bajo + con_valor)}
    return salida


def ambiente_entre_subestaciones(datos):
    """Correlacion diaria del ambiente de S1 (registro de T1) y S2 (T3)."""
    a1 = {f["fecha"]: f["ambiente_c"] for f in datos["T1"]
          if f.get("ambiente_c") is not None}
    a2 = {f["fecha"]: f["ambiente_c"] for f in datos["T3"]
          if f.get("ambiente_c") is not None}
    comunes = sorted(set(a1) & set(a2))
    x = np.array([a1[d] for d in comunes])
    y = np.array([a2[d] for d in comunes])
    return {"dias_comunes": len(comunes),
            "correlacion": float(np.corrcoef(x, y)[0, 1])}


def factor_comun_s1(datos, minimo_dias_mes=3):
    """Parte del aceite de S1 que el ambiente registrado no explica.

    T1 y T2 estan en la misma subestacion y comparten el registro de ambiente.
    Para cada una se ajusta aceite = a + b*carga + c*ambiente sobre los dias
    que tienen las dos, y se toma el residuo. Si los residuos de las dos
    maquinas suben y bajan juntos, hay algo que actua sobre ambas y que el
    ambiente registrado no recoge. Su media mensual se compara despues con el
    ambiente registrado en S1 y con el de S2.
    """
    def completa(f):
        return all(f.get(k) is not None
                   for k in ("aceite_max_c", "carga_pu", "ambiente_c"))
    por_fecha = {et: {f["fecha"]: f for f in datos[et] if completa(f)}
                 for et in ("T1", "T2")}
    fechas = sorted(set(por_fecha["T1"]) & set(por_fecha["T2"]))
    residuos = []
    for et in ("T1", "T2"):
        filas = [por_fecha[et][d] for d in fechas]
        y = np.array([f["aceite_max_c"] for f in filas])
        A = np.column_stack([np.ones(len(filas)),
                             [f["carga_pu"] for f in filas],
                             [f["ambiente_c"] for f in filas]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        residuos.append(y - A @ coef)
    comun = (residuos[0] + residuos[1]) / 2.0

    def por_mes(pares):
        meses = {}
        for fecha, valor in pares:
            meses.setdefault(fecha[:7], []).append(valor)
        return meses
    m_comun = por_mes(zip(fechas, comun))
    m_s1 = por_mes((f["fecha"], f["ambiente_c"]) for f in datos["T1"]
                   if f.get("ambiente_c") is not None)
    m_s2 = por_mes((f["fecha"], f["ambiente_c"]) for f in datos["T3"]
                   if f.get("ambiente_c") is not None)
    meses = [m for m in sorted(m_comun)
             if len(m_comun[m]) >= minimo_dias_mes and m in m_s1 and m in m_s2]
    x = [np.mean(m_comun[m]) for m in meses]
    a1 = [np.mean(m_s1[m]) for m in meses]
    a2 = [np.mean(m_s2[m]) for m in meses]
    return {"dias_comunes": len(fechas),
            "correlacion_residuos_T1_T2": float(np.corrcoef(*residuos)[0, 1]),
            "meses": len(meses),
            "correlacion_mensual_con_ambiente_S1": float(np.corrcoef(x, a1)[0, 1]),
            "correlacion_mensual_con_ambiente_S2": float(np.corrcoef(x, a2)[0, 1])}


def resumen_mensual(filas):
    por_mes = defaultdict(list)
    for f in filas:
        por_mes[f["fecha"][:7]].append(f)
    tabla = []
    for mes in sorted(por_mes):
        g = por_mes[mes]
        carga = [f["carga_pu"] for f in g if f["carga_pu"] is not None]
        amb = [f["ambiente_c"] for f in g if f["ambiente_c"] is not None]
        maxs = [f["aceite_max_c"] for f in g if f["aceite_max_c"] is not None]
        tabla.append({
            "mes": mes, "dias": len(g),
            "carga_media": sum(carga) / len(carga) if carga else None,
            "carga_maxima": max(carga) if carga else None,
            "ambiente_medio": sum(amb) / len(amb) if amb else None,
            "aceite_maximo_medio": sum(maxs) / len(maxs) if maxs else None,
            "dias_con_maximo": len(maxs)})
    return tabla


# Comprobacion de la anonimizacion: cada celda de los archivos publicados debe
# ser una etiqueta del esquema, un numero, una fecha normalizada o estar vacia.
ETIQUETAS_PERMITIDAS = {"T1", "T2", "T3", "S1", "S2"}
NUMERO = re.compile(r"^[-+]?\d*\.?\d+([eE][-+]?\d+)?$")
FECHA_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def celda_admisible(valor):
    t = (valor or "").strip()
    return (t == "" or t in ETIQUETAS_PERMITIDAS
            or bool(NUMERO.match(t)) or bool(FECHA_ISO.match(t)))


def _recorrer_json(nodo, ruta, revisadas, hallazgos):
    # Las claves del archivo de incidencias son nombres de campo fijados por
    # este programa; lo que se revisa son los valores de texto que contiene.
    if isinstance(nodo, dict):
        for clave, valor in nodo.items():
            revisadas = _recorrer_json(valor, "%s.%s" % (ruta, clave),
                                       revisadas, hallazgos)
    elif isinstance(nodo, list):
        for i, valor in enumerate(nodo):
            revisadas = _recorrer_json(valor, "%s[%d]" % (ruta, i),
                                       revisadas, hallazgos)
    else:
        revisadas += 1
        if isinstance(nodo, str) and not celda_admisible(nodo):
            hallazgos.append(["incidencias.json", ruta, nodo])
    return revisadas


def verificar_anonimizacion(carpeta_datos):
    revisadas, hallazgos = 0, []
    for et in MAQUINAS:
        ruta = os.path.join(carpeta_datos, "%s.csv" % et)
        with open(ruta, encoding="utf-8") as fh:
            for fila_n, fila in enumerate(csv.DictReader(fh), start=2):
                for columna, valor in fila.items():
                    revisadas += 1
                    if not celda_admisible(valor):
                        hallazgos.append([et, fila_n, columna])
    # El archivo de incidencias se publica junto a los conjuntos, de modo que
    # entra en la misma comprobacion.
    ruta_inc = os.path.join(carpeta_datos, "incidencias.json")
    if os.path.exists(ruta_inc):
        with open(ruta_inc, encoding="utf-8") as fh:
            revisadas = _recorrer_json(json.load(fh), "incidencias",
                                       revisadas, hallazgos)
    return {"celdas_revisadas": revisadas, "hallazgos": hallazgos}


def estado_de_condicion(datos):
    maquinas = {}
    for et in MAQUINAS:
        maquinas[et] = {
            "subestacion": datos[et][0]["subestacion"],
            "dias_en_el_conjunto": len(datos[et]),
            "maximo_frente_a_ronda": maximo_frente_a_ronda(datos[et]),
            "gradiente_bobinado_aceite": gradiente_bobinado(datos[et]),
            "deriva_del_residuo": deriva(datos[et]),
            "cambiador_de_tomas": ritmo_cambiador(datos[et]),
            "regresion_con_ambiente": regresion_con_ambiente(datos[et]),
            "resumen_mensual": resumen_mensual(datos[et]),
        }
    return {"transformadores": maquinas,
            "gases_T1": gases_disueltos(datos["T1"]),
            "ambiente_entre_subestaciones": ambiente_entre_subestaciones(datos),
            "factor_comun_S1": factor_comun_s1(datos),
            "umbrales_maniobras": list(UMBRALES_MANIOBRAS)}


# -------------------------------------------------------------------- figuras

def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.edgecolor": TINTA_SUAVE, "axes.labelcolor": TINTA,
        "axes.linewidth": 0.7, "axes.titlesize": 10,
        "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.4, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def limpiar(ax, rejilla="y"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if rejilla:
        ax.grid(axis=rejilla, color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)


def figura_elevacion(datos, resultados, destino):
    fig, ejes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
    for ax, et in zip(ejes, MAQUINAS):
        _, carga, aceite, amb, _, _ = vectores(datos[et], "aceite_max_c")
        delta = resultados[et]["maximo"]["con_placa"]["delta_r"]
        ax.scatter(carga, aceite - amb, s=9, color=COLOR[et], alpha=0.45,
                   edgecolors="none", marker=MARCA[et])
        rej = np.linspace(max(carga.min() - 0.03, 0.01), carga.max() + 0.03, 120)
        ax.plot(rej, delta * factor_carga(rej, 1.0), color=TINTA, linewidth=1.5,
                label=r"$\Delta\theta_R$=%.0f $^\circ$C" % delta)
        ax.set_title(et, color=COLOR[et])
        ax.set_xlabel("carga (pu)")
        ax.legend(loc="upper left", fontsize=7)
        limpiar(ax)
    ejes[0].set_ylabel("elevación sobre\nel ambiente ($^\\circ$C)")
    fig.suptitle("Elevación observada y curva normativa con la placa fijada",
                 y=1.04, fontsize=10)
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def figura_curva_sr(resultados, destino):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for et in MAQUINAS:
        curva = np.array(resultados[et]["maximo"]["curva_sr"])
        ax.plot(curva[:, 0], curva[:, 1], TRAZO[et], color=COLOR[et], label=et)
    ax.axvline(1.0, color=TINTA, linewidth=1.0, linestyle=":")
    ax.set_xlabel("potencia nominal supuesta (pu de la placa)")
    ax.set_ylabel("error absoluto medio ($^\\circ$C)")
    ax.set_xlim(0.3, 3.0)
    ax.legend(ncol=3)
    limpiar(ax)
    ax.set_title("La curva de error es casi plana en T1 y T2", fontsize=9.5)
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def figura_validacion(resultados, destino):
    modelos = ["placa", "libre", "lineal", "media"]
    etiquetas = {"placa": "Normativo con placa (1 par.)",
                 "libre": "Normativo libre (2 par.)",
                 "lineal": "Lineal (2 par.)",
                 "media": "Elevación constante (1 par.)"}
    colores = {"placa": COLOR["T1"], "libre": COLOR["T3"],
               "lineal": COLOR["T2"], "media": TINTA_SUAVE}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(MAQUINAS))
    ancho = 0.2
    for j, m in enumerate(modelos):
        valores = [resultados[et]["maximo"]["cv"][m] for et in MAQUINAS]
        ax.bar(x + (j - 1.5) * ancho, valores, ancho * 0.88,
               color=colores[m], label=etiquetas[m])
    ax.set_xticks(x)
    ax.set_xticklabels(MAQUINAS)
    ax.set_ylabel("MAE fuera de muestra ($^\\circ$C)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    limpiar(ax)
    ax.set_title("Error fuera de muestra, dejando un mes fuera", fontsize=9.5)
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def figura_residuos(datos, resultados, destino):
    fig, ejes = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
    for ax, et in zip(ejes, MAQUINAS):
        _, carga, aceite, amb, _, _ = vectores(datos[et], "aceite_max_c")
        delta = resultados[et]["maximo"]["con_placa"]["delta_r"]
        resid = (aceite - amb) - delta * factor_carga(carga, 1.0)
        ax.scatter(amb, resid, s=9, color=COLOR[et], alpha=0.45,
                   edgecolors="none", marker=MARCA[et])
        ax.axhline(0.0, color=TINTA, linewidth=0.9)
        coef = np.polyfit(amb, resid, 1)
        xs = np.linspace(amb.min(), amb.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color=TINTA, linewidth=1.1,
                label="pendiente %.2f" % coef[0])
        ax.set_title(et, color=COLOR[et])
        ax.set_xlabel("ambiente ($^\\circ$C)")
        ax.legend(loc="upper right", fontsize=7)
        limpiar(ax)
    ejes[0].set_ylabel("residuo ($^\\circ$C)")
    fig.suptitle("Residuos frente a la temperatura ambiente", y=1.05,
                 fontsize=10)
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def figura_sintetico(sintetico, observadas, destino):
    celdas = sintetico["celdas"]
    recorridos = sorted({c["recorrido"] for c in celdas})
    errores = sorted({c["error_ambiente"] for c in celdas})
    paleta = [COLOR["T1"], COLOR["T2"], COLOR["T3"], "#4a7a20", TINTA,
              TINTA_SUAVE, "#8c4a00"]
    marcas = ["o", "s", "^", "v", "D", "P", "X"]

    fig, ejes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"wspace": 0.55})
    for i, e in enumerate(errores):
        ys = [next(c["error_sr_mediana_pct"] for c in celdas
                   if c["recorrido"] == r and c["error_ambiente"] == e)
              for r in recorridos]
        ejes[0].plot(recorridos, ys, marcas[i % len(marcas)] + "-",
                     color=paleta[i % len(paleta)], markersize=3.5,
                     linewidth=1.1, label="%.0f $^\\circ$C" % e)
    ejes[0].set_xlabel("recorrido de carga (pu)")
    ejes[0].set_ylabel("error de recuperación\nde $S_R$ (%, mediana)")
    ejes[0].legend(title="error del ambiente", ncol=2, fontsize=7,
                   title_fontsize=7)
    limpiar(ejes[0])

    ys = [np.median([c["pendiente_residuo_mediana"] for c in celdas
                     if c["error_ambiente"] == e]) for e in errores]
    ejes[1].plot(errores, ys, "o-", color=TINTA, markersize=4,
                 linewidth=1.4, label="simulación")
    mas_baja = min(observadas, key=observadas.get) if observadas else None
    for et, valor in observadas.items():
        ejes[1].axhline(valor, color=COLOR[et], linestyle=TRAZO[et],
                        linewidth=1.1)
        ejes[1].annotate(et, xy=(0.15, valor),
                         ha="left", va="top" if et == mas_baja else "bottom",
                         fontsize=7.5, color=COLOR[et])
    ejes[1].set_xlabel("error de medida del ambiente ($^\\circ$C)")
    ejes[1].set_ylabel("pendiente del residuo\nfrente al ambiente")
    ejes[1].set_ylim(-1.15, 0.45)
    ejes[1].legend(loc="center right", fontsize=7)
    limpiar(ejes[1])

    fig.suptitle("Identificabilidad simulada y diagnóstico del registro "
                 "de ambiente", y=1.05, fontsize=9.5)
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def guardar(fig, destino):
    fig.savefig(destino)
    fig.savefig(destino[:-4] + ".png", dpi=200)
    plt.close(fig)


def figura_cobertura(datos, destino):
    fig, ejes = plt.subplots(3, 1, figsize=(6.3, 4.4), sharex=True)
    meses = sorted({f["fecha"][:7] for fs in datos.values() for f in fs})
    x = np.arange(len(meses))
    indice = {m: i for i, m in enumerate(meses)}
    for ax, et in zip(ejes, MAQUINAS):
        con_max, con_ronda = np.zeros(len(meses)), np.zeros(len(meses))
        for f in datos[et]:
            i = indice[f["fecha"][:7]]
            if f["aceite_max_c"] is not None:
                con_max[i] += 1
            elif f["aceite_ronda_c"] is not None:
                con_ronda[i] += 1
        ax.bar(x, con_max, width=0.72, color=COLOR[et],
               label="con máximo diario")
        ax.bar(x, con_ronda, width=0.72, bottom=con_max, color="white",
               edgecolor=COLOR[et], linewidth=0.7, hatch="///",
               label="solo lectura de ronda")
        ax.set_ylabel("días  %s" % et)
        ax.set_ylim(0, 32)
        limpiar(ax)
        if et == "T1":
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ejes[-1].set_xticks(x[::2])
    ejes[-1].set_xticklabels([meses[i] for i in range(0, len(meses), 2)],
                             rotation=60, ha="right")
    fig.suptitle("Días utilizables por mes y tipo de lectura de temperatura",
                 y=1.04, fontsize=10)
    guardar(fig, destino)


def figura_distribucion_carga(datos, destino):
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    series = [vectores(datos[et], "aceite_max_c")[1] for et in MAQUINAS]
    # matplotlib 3.10 cambio 'vert' por 'orientation'; se usa el que exista.
    version = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    horizontal = ({"orientation": "horizontal"} if version >= (3, 10)
                  else {"vert": False})
    partes = ax.boxplot(series, **horizontal, widths=0.55, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.4),
                        flierprops=dict(marker=".", markersize=3,
                                        markerfacecolor=TINTA_SUAVE,
                                        markeredgecolor="none", alpha=0.6),
                        whiskerprops=dict(color=TINTA_SUAVE, linewidth=0.8),
                        capprops=dict(color=TINTA_SUAVE, linewidth=0.8))
    for caja, et in zip(partes["boxes"], MAQUINAS):
        caja.set_facecolor(COLOR[et])
        caja.set_edgecolor("none")
    for i, s in enumerate(series, start=1):
        recorrido = np.percentile(s, 90) - np.percentile(s, 10)
        ax.annotate("recorrido p10-p90: %.2f pu" % recorrido, xy=(1.04, i),
                    va="center", ha="left", fontsize=7.5, color=TINTA)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(MAQUINAS)
    ax.set_xlabel("carga diaria máxima, en por unidad de la placa")
    ax.set_xlim(0, 1.62)
    limpiar(ax, rejilla="x")
    ax.set_title("La carga de T3 recorre un rango más amplio que la de T1 y T2")
    guardar(fig, destino)


def figura_maximo_ronda(datos, destino):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for et in MAQUINAS:
        pares = [(f["aceite_ronda_c"], f["aceite_max_c"]) for f in datos[et]
                 if f["aceite_ronda_c"] is not None
                 and f["aceite_max_c"] is not None]
        if not pares:
            continue
        a = np.array([p[0] for p in pares])
        b = np.array([p[1] for p in pares])
        ax.scatter(a, b, s=10, color=COLOR[et], alpha=0.5, edgecolors="none",
                   marker=MARCA[et],
                   label="%s (mediana +%.0f $^\\circ$C)" % (et, np.median(b - a)))
    lim = [25, 80]
    ax.plot(lim, lim, color=TINTA, linewidth=1.0, linestyle=":")
    ax.annotate("igualdad", xy=(lim[1] - 12, lim[1] - 10), fontsize=7.5,
                color=TINTA_SUAVE, rotation=38)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("aceite, lectura de ronda ($^\\circ$C)")
    ax.set_ylabel("aceite, máximo diario ($^\\circ$C)")
    ax.legend(loc="upper left", fontsize=7.5)
    limpiar(ax, rejilla="both")
    ax.set_title("El máximo diario supera sistemáticamente\nla lectura de ronda",
                 fontsize=9.5)
    guardar(fig, destino)


def figura_gases(datos, destino):
    verde = "#4a7a20"
    fig, ejes = plt.subplots(2, 1, figsize=(6.3, 4.0), sharex=True)
    filas = sorted((f for f in datos["T1"] if f.get("gas_H2") is not None),
                   key=lambda f: f["fecha"])
    fechas = [f["fecha"] for f in filas]
    x = np.arange(len(fechas))
    paneles = (
        (ejes[0], (("gas_H2", "H$_2$", COLOR["T1"], "o"),
                   ("gas_CH4", "CH$_4$", COLOR["T2"], "s"),
                   ("gas_C2H4", "C$_2$H$_4$", COLOR["T3"], "^"),
                   ("gas_C2H6", "C$_2$H$_6$", verde, "v"))),
        (ejes[1], (("gas_CO", "CO", COLOR["T1"], "o"),
                   ("gas_CO2", "CO$_2$", COLOR["T2"], "s"))),
    )
    for ax, series in paneles:
        for clave, etiqueta, color, marca in series:
            y = np.array([f.get(clave) if f.get(clave) is not None else np.nan
                          for f in filas], dtype=float)
            ax.plot(x, y, marca + "-", color=color, markersize=2.6,
                    linewidth=0.9, label=etiqueta, alpha=0.9)
        ax.set_ylabel("ppm")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        limpiar(ax)
    paso = max(len(x) // 10, 1)
    ejes[1].set_xticks(x[::paso])
    ejes[1].set_xticklabels([fechas[i][:7] for i in range(0, len(x), paso)],
                            rotation=60, ha="right")
    ejes[0].set_title("Gases disueltos en el aceite de T1", fontsize=10)
    guardar(fig, destino)


def figura_envejecimiento(datos, destino):
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for et in MAQUINAS:
        bob = [f["bobinado_max_c"] for f in datos[et]
               if f.get("bobinado_max_c") is not None]
        if len(bob) < 20:
            continue
        faa = factor_envejecimiento(np.array(bob))
        orden = np.sort(faa)[::-1]
        ax.plot(np.arange(len(orden)) / len(orden) * 100.0, orden, TRAZO[et],
                color=COLOR[et],
                label="%s (%.2f días/año)" % (et, faa.mean() * 365.0))
    ax.axhline(1.0, color=TINTA, linewidth=1.0, linestyle=":")
    ax.annotate("envejecimiento nominal, 110 $^\\circ$C", xy=(52, 1.08),
                fontsize=7.5, color=TINTA)
    ax.set_yscale("log")
    ax.set_xlabel("porcentaje de días por encima del valor")
    ax.set_ylabel("factor de envejecimiento")
    ax.legend(loc="lower left")
    limpiar(ax)
    ax.set_title("Todos los días quedan muy por debajo del envejecimiento "
                 "nominal", fontsize=9.5)
    guardar(fig, destino)


def figura_gradiente(datos, destino):
    fig, ejes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    for ax, et in zip(ejes, MAQUINAS):
        pares = [(f["carga_pu"], f["bobinado_max_c"] - f["aceite_max_c"])
                 for f in datos[et]
                 if f.get("aceite_max_c") is not None
                 and f.get("bobinado_max_c") is not None
                 and f.get("carga_pu") is not None]
        if len(pares) < 20:
            continue
        k = np.array([p[0] for p in pares])
        g = np.array([p[1] for p in pares])
        ax.scatter(k ** 2, g, s=9, color=COLOR[et], alpha=0.45,
                   edgecolors="none", marker=MARCA[et])
        ax.axhline(0.0, color=TINTA, linewidth=1.0)
        ax.set_title("%s  (%.0f%% invertidos)" % (et, 100.0 * np.mean(g < 0)),
                     color=COLOR[et], fontsize=9)
        ax.set_xlabel("$K^2$ (pu$^2$)")
        limpiar(ax)
    ejes[0].set_ylabel("bobinado $-$ aceite ($^\\circ$C)")
    fig.suptitle("El gradiente no puede ser negativo, y en S1 lo es con "
                 "frecuencia", y=1.06, fontsize=9.5)
    guardar(fig, destino)


def figura_deriva(datos, destino):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for et in MAQUINAS:
        u, t, resid = residuo_en_el_tiempo(datos[et])
        por_mes = defaultdict(list)
        for f, tt, r in zip(u, t, resid):
            por_mes[f["fecha"][:7]].append((tt, r))
        puntos = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]))
                        for v in por_mes.values())
        ax.plot([p[0] for p in puntos], [p[1] for p in puntos], MARCA[et],
                color=COLOR[et], markersize=3.2, alpha=0.6, linestyle="none")
        coef = np.polyfit(t, resid, 1)
        xs = np.linspace(t.min(), t.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), TRAZO[et], color=COLOR[et],
                linewidth=1.5,
                label="%s: %+.2f $^\\circ$C/mes" % (et, coef[0]))
    ax.axhline(0.0, color=TINTA, linewidth=0.9)
    ax.set_xlabel("meses desde el inicio del periodo")
    ax.set_ylabel("residuo medio mensual ($^\\circ$C)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    limpiar(ax)
    ax.set_title("Las tres máquinas se enfrían respecto de lo que el modelo "
                 "predice", fontsize=9.5)
    guardar(fig, destino)


# --------------------------------------------------------------------- flujo

def analizar_maquina(filas):
    """Ejecuta el analisis completo de un transformador."""
    salida = {}
    for variante, campo in VARIANTES.items():
        utiles, carga, aceite, amb, meses, periodos = vectores(filas, campo)
        if len(utiles) < 30:
            continue

        placa = ajustar_con_placa(carga, aceite, amb)
        libre = ajustar_libre(carga, aceite, amb)
        bajo, alto = intervalo_equivalencia(libre)

        resid = (aceite - amb) - placa["delta_r"] * factor_carga(carga, 1.0)
        pendiente_resid = float(np.polyfit(amb, resid, 1)[0])

        completos = [f for f in utiles if f["horas"] >= COBERTURA_COMPLETA]
        robustez = None
        if len(completos) >= 30:
            _, c2, a2, m2, _, _ = vectores(completos, campo)
            robustez = {"dias": len(completos),
                        "delta_r": ajustar_con_placa(c2, a2, m2)["delta_r"]}

        envejecimiento = None
        bob = [f[BOBINADO[variante]] for f in utiles
               if f.get(BOBINADO[variante]) is not None]
        if len(bob) >= 30:
            faa = factor_envejecimiento(np.array(bob))
            envejecimiento = {
                "dias": len(bob),
                "bobinado_medio_c": float(np.mean(bob)),
                "bobinado_maximo_c": float(np.max(bob)),
                "dias_equivalentes_por_anio": float(np.mean(faa) * 365.0),
            }

        salida[variante] = {
            "dias": len(utiles),
            "meses": len(set(meses)),
            "carga_pu_mediana": float(np.median(carga)),
            "carga_pu_maxima": float(np.max(carga)),
            "carga_pu_p10": float(np.percentile(carga, 10)),
            "carga_pu_p90": float(np.percentile(carga, 90)),
            "correlacion_aceite_ambiente": float(np.corrcoef(aceite, amb)[0, 1]),
            "con_placa": placa,
            "exceso_de_la_placa_sobre_el_minimo_pct":
                (placa["mae"] / libre["mae"] - 1.0) * 100.0,
            "libre": {"sr_pu": libre["sr_pu"], "delta_r": libre["delta_r"],
                      "mae": libre["mae"], "en_el_borde": libre["en_el_borde"],
                      "intervalo_equivalencia": [bajo, alto],
                      "contiene_la_placa": bool(bajo <= 1.0 <= alto)},
            "lineal": ajustar_lineal(carga, aceite, amb),
            "media": ajustar_media(aceite, amb),
            "cv": validacion_cruzada(carga, aceite, amb, meses),
            "temporal": validacion_temporal(carga, aceite, amb, periodos),
            "pendiente_residuo_ambiente": pendiente_resid,
            "robustez_cobertura_completa": robustez,
            "envejecimiento": envejecimiento,
            "curva_sr": [[sr, mae] for sr, _, mae in libre["curva"]],
        }
    return salida


def informe(resultados):
    """Imprime en pantalla las cifras principales."""
    print()
    print("=" * 72)
    print("RESULTADOS")
    print("=" * 72)
    for et in MAQUINAS:
        if et not in resultados:
            continue
        print()
        print("--- %s ---" % et)
        for variante in ("maximo", "ronda"):
            r = resultados[et].get(variante)
            if not r:
                continue
            p, l, cv = r["con_placa"], r["libre"], r["cv"]
            mejora = (1.0 - cv["placa"] / cv["media"]) * 100.0
            print("  variante %-7s  %d dias, %d meses" % (variante, r["dias"],
                                                          r["meses"]))
            print("     con la placa fijada   deltaR = %5.1f C   MAE = %.2f C"
                  % (p["delta_r"], p["mae"]))
            print("     con los dos libres    Sr = %.2f pu   equiv [%.2f, %.2f]"
                  "   contiene la placa: %s"
                  % (l["sr_pu"], l["intervalo_equivalencia"][0],
                     l["intervalo_equivalencia"][1],
                     "si" if l["contiene_la_placa"] else "no"))
            print("     fuera de muestra      placa %.2f | libre %.2f | "
                  "lineal %.2f | constante %.2f  ->  %+.1f%%"
                  % (cv["placa"], cv["libre"], cv["lineal"], cv["media"],
                     mejora))
            if r["temporal"]:
                print("     validacion temporal   placa %.2f | lineal %.2f"
                      % (r["temporal"]["mae_con_placa"],
                         r["temporal"]["mae_lineal"]))
            print("     residuo vs ambiente   pendiente %+.2f"
                  % r["pendiente_residuo_ambiente"])
            if r["envejecimiento"]:
                print("     envejecimiento        %.2f dias equivalentes por anio"
                      % r["envejecimiento"]["dias_equivalentes_por_anio"])


def informe_sintetico(sintetico):
    print()
    print("=" * 72)
    print("EXPERIMENTO DE IDENTIFICABILIDAD")
    print("=" * 72)
    print("%-11s %-9s %13s %13s %11s" % ("recorrido", "err.amb",
                                          "err|Sr| p50", "err|Sr| p90",
                                          "pend.resid"))
    for c in sintetico["celdas"]:
        print("%-11.2f %-9.1f %12.1f%% %12.1f%% %11.2f"
              % (c["recorrido"], c["error_ambiente"],
                 c["error_sr_mediana_pct"], c["error_sr_p90_pct"],
                 c["pendiente_residuo_mediana"]))


def informe_condicion(condicion):
    print()
    print("=== Estado de condicion ===")
    for et, c in condicion["transformadores"].items():
        print("--- %s ---" % et)
        m = c["maximo_frente_a_ronda"]
        if m:
            print("  maximo - ronda       mediana %+.1f C, maximo mayor en %.0f%% de %d dias"
                  % (m["diferencia_mediana"], 100 * m["proporcion_maximo_mayor"],
                     m["dias"]))
        g = c["gradiente_bobinado_aceite"]
        if g:
            print("  bobinado - aceite    mediana %.1f C, invertido en %d dias (%.0f%%)"
                  % (g["mediana"], g["invertidos"],
                     100 * g["proporcion_invertidos"]))
        d = c["deriva_del_residuo"]
        print("  deriva del residuo   %+.2f C/mes  (t = %.1f)"
              % (d["grados_por_mes"], d["estadistico_t"]))
        r = c["regresion_con_ambiente"]
        print("  aceite y ambiente    coeficiente %.2f con el maximo, %.2f con la ronda (la norma supone 1)"
              % (r["coeficiente_ambiente"], r["coeficiente_ambiente_ronda"]))
        o = c["cambiador_de_tomas"]
        if o:
            print("  cambiador de tomas   %.0f maniobras/anio, %d retrocesos del contador en %d dias"
                  % (o["maniobras_por_anio"], o["retrocesos_del_contador"],
                     o["dias"]))
    v = condicion["anonimizacion"]
    print("--- anonimizacion: %d celdas revisadas, %d hallazgos"
          % (v["celdas_revisadas"], len(v["hallazgos"])))
    a = condicion["ambiente_entre_subestaciones"]
    print("--- ambiente S1 frente a S2: correlacion %.2f en %d dias comunes"
          % (a["correlacion"], a["dias_comunes"]))
    f = condicion["factor_comun_S1"]
    print("--- residuos de T1 y T2: correlacion %.2f en %d dias; su media mensual "
          "frente al ambiente de S1 %.2f y de S2 %.2f (%d meses)"
          % (f["correlacion_residuos_T1_T2"], f["dias_comunes"],
             f["correlacion_mensual_con_ambiente_S1"],
             f["correlacion_mensual_con_ambiente_S2"], f["meses"]))
    g = condicion["gases_T1"]
    if "H2" in g:
        print("--- gases en T1: H2 mediana %.0f ppm, maximo %.0f ppm"
              % (g["H2"]["mediana"], g["H2"]["maximo"]))
    if "C2H2_bajo_el_limite" in g:
        a = g["C2H2_bajo_el_limite"]
        print("    acetileno: %d lecturas, %d bajo el limite de deteccion (%.0f%%), "
              "el resto en cero" % (a["lecturas_de_acetileno"], a["lecturas"],
                                    100 * a["proporcion"]))


def main(carpeta_datos, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)
    carpeta_figuras = os.path.join(carpeta_salida, "figuras")
    os.makedirs(carpeta_figuras, exist_ok=True)

    faltan = [et for et in MAQUINAS
              if not os.path.exists(os.path.join(carpeta_datos, "%s.csv" % et))]
    if faltan:
        sys.exit("No se encuentran %s en la carpeta:\n   %s"
                 % (", ".join(et + ".csv" for et in faltan),
                    os.path.abspath(carpeta_datos)))

    print("Datos:  %s" % os.path.abspath(carpeta_datos))
    print("Salida: %s" % os.path.abspath(carpeta_salida))
    print()
    print("[1/4] Ajuste, validacion y condicion de cada transformador",
          flush=True)
    datos, resultados = {}, {}
    for et in MAQUINAS:
        print("   %s" % et, flush=True)
        datos[et] = leer_conjunto(os.path.join(carpeta_datos, "%s.csv" % et))
        resultados[et] = analizar_maquina(datos[et])
    condicion = estado_de_condicion(datos)
    condicion["anonimizacion"] = verificar_anonimizacion(carpeta_datos)

    print("[2/4] Experimento de identificabilidad (es la parte que mas tarda)",
          flush=True)
    sintetico = experimento_identificabilidad()

    observadas = {et: resultados[et]["maximo"]["pendiente_residuo_ambiente"]
                  for et in resultados if "maximo" in resultados[et]}

    print("[3/4] Figuras", flush=True)
    estilo()
    d = lambda nombre: os.path.join(carpeta_figuras, nombre)
    figura_elevacion(datos, resultados, d("elevacion_vs_carga.pdf"))
    figura_curva_sr(resultados, d("curva_sr.pdf"))
    figura_validacion(resultados, d("validacion.pdf"))
    figura_residuos(datos, resultados, d("residuos.pdf"))
    figura_sintetico(sintetico, observadas, d("sintetico.pdf"))
    figura_cobertura(datos, d("cobertura.pdf"))
    figura_distribucion_carga(datos, d("distribucion_carga.pdf"))
    figura_maximo_ronda(datos, d("max_vs_ronda.pdf"))
    figura_gradiente(datos, d("gradiente.pdf"))
    figura_deriva(datos, d("deriva.pdf"))
    figura_envejecimiento(datos, d("envejecimiento.pdf"))
    figura_gases(datos, d("gases.pdf"))

    with open(os.path.join(carpeta_salida, "resultados.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"R": R_NOMINAL, "n": N_NOMINAL,
                   "transformadores": resultados}, fh, indent=2,
                  ensure_ascii=False)
    with open(os.path.join(carpeta_salida, "sintetico.json"), "w",
              encoding="utf-8") as fh:
        json.dump(sintetico, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(carpeta_salida, "condicion.json"), "w",
              encoding="utf-8") as fh:
        json.dump(condicion, fh, indent=2, ensure_ascii=False)

    print("[4/4] Resultados", flush=True)
    informe(resultados)
    informe_sintetico(sintetico)
    informe_condicion(condicion)
    print()
    print("Escrito en %s" % os.path.abspath(carpeta_salida))


def carpeta_por_omision():
    """Busca datos/ junto al programa y, si no esta, un nivel arriba.
    Asi funciona igual si el programa esta en la raiz o dentro de src/,
    y no depende de la carpeta desde la que se ejecute."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    for base in (aqui, os.path.dirname(aqui)):
        if os.path.isdir(os.path.join(base, "datos")):
            return os.path.join(base, "datos"), os.path.join(base, "salida")
    return os.path.join(aqui, "datos"), os.path.join(aqui, "salida")


if __name__ == "__main__":
    datos_omision, salida_omision = carpeta_por_omision()
    entrada = sys.argv[1] if len(sys.argv) > 1 else datos_omision
    salida = sys.argv[2] if len(sys.argv) > 2 else salida_omision
    main(entrada, salida)
