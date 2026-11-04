"""Tabla RFM, puntuaciones 1-5 y segmentos con nombre."""

from __future__ import annotations

import re

import pandas as pd

# Mapa clásico de segmentos según las puntuaciones de recencia (R) y frecuencia (F).
# Se evalúa en orden: la primera regla que encaja gana.
SEGMENTOS = [
    ("Campeones", r"5[4-5]"),
    ("Leales", r"[3-4][4-5]"),
    ("Potencialmente leales", r"[4-5][2-3]"),
    ("Nuevos", r"51"),
    ("Prometedores", r"41"),
    ("Necesitan atención", r"33"),
    ("A punto de dormirse", r"3[1-2]"),
    ("No se pueden perder", r"[1-2]5"),
    ("En riesgo", r"[1-2][3-4]"),
    ("Hibernando", r"[1-2][1-2]"),
]

ACCIONES = {
    "Campeones": "Premiar: acceso anticipado, programa VIP. Pueden ser embajadores.",
    "Leales": "Venta cruzada y de mayor valor; pedirles reseñas.",
    "Potencialmente leales": "Programa de fidelización y recomendaciones personalizadas.",
    "Nuevos": "Buena bienvenida: guía de producto y oferta para la 2.ª compra.",
    "Prometedores": "Dar a conocer el catálogo; ofertas de entrada.",
    "Necesitan atención": "Ofertas por tiempo limitado basadas en su historial.",
    "A punto de dormirse": "Reactivar con contenido y descuentos antes de perderlos.",
    "No se pueden perder": "Contacto directo y personal: eran de los mejores y se han ido.",
    "En riesgo": "Campaña de recuperación con incentivo fuerte.",
    "Hibernando": "Reactivación barata (email); si no responden, no invertir más.",
}


def tabla_rfm(ventas: pd.DataFrame, devoluciones: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Recencia (días desde la última compra), frecuencia (facturas) e importe neto de devoluciones."""
    v = ventas[ventas["fecha"] < fecha_corte]
    d = devoluciones[devoluciones["fecha"] < fecha_corte]
    rfm = v.groupby("cliente").agg(
        ultima_compra=("fecha", "max"),
        primera_compra=("fecha", "min"),
        frecuencia=("factura", "nunique"),
        importe_bruto=("importe", "sum"),
        pais=("pais", "first"),
    )
    rfm["devuelto"] = d.groupby("cliente")["importe"].sum().reindex(rfm.index).fillna(0)
    rfm["importe"] = (rfm["importe_bruto"] - rfm["devuelto"]).clip(lower=0)
    rfm["recencia"] = (fecha_corte.normalize() - rfm["ultima_compra"].dt.normalize()).dt.days
    rfm["antiguedad"] = (fecha_corte.normalize() - rfm["primera_compra"].dt.normalize()).dt.days
    return rfm


def puntuar(rfm: pd.DataFrame) -> pd.DataFrame:
    """Puntuaciones 1-5 por quintiles. Se puntúa sobre el ranking para que los empates
    (muchos clientes con 1 sola compra) no rompan los cortes."""
    r = rfm.copy()
    r["R"] = pd.qcut(r["recencia"].rank(method="first", ascending=False), 5, labels=range(1, 6)).astype(int)
    r["F"] = pd.qcut(r["frecuencia"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
    r["M"] = pd.qcut(r["importe"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
    r["RFM"] = r["R"].astype(str) + r["F"].astype(str) + r["M"].astype(str)
    r["segmento"] = [segmento(rr, ff) for rr, ff in zip(r["R"], r["F"])]
    return r


def segmento(r: int, f: int) -> str:
    rf = f"{r}{f}"
    for nombre, patron in SEGMENTOS:
        if re.fullmatch(patron, rf):
            return nombre
    raise ValueError(f"Sin segmento para R={r} F={f}")


def resumen_segmentos(r: pd.DataFrame) -> pd.DataFrame:
    total_clientes, total_importe = len(r), r["importe"].sum()
    s = r.groupby("segmento").agg(
        clientes=("recencia", "size"),
        recencia_mediana=("recencia", "median"),
        frecuencia_media=("frecuencia", "mean"),
        importe_medio=("importe", "mean"),
        importe_total=("importe", "sum"),
    )
    s["pct_clientes"] = s["clientes"] / total_clientes
    s["pct_importe"] = s["importe_total"] / total_importe
    s["accion"] = s.index.map(ACCIONES)
    orden = [n for n, _ in SEGMENTOS]
    return s.reindex([o for o in orden if o in s.index])
