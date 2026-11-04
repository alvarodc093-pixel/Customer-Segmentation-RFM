"""Descarga y limpieza de Online Retail II (UCI, CC BY 4.0)."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "data" / "online_retail_II.xlsx"
PARQUET = RAIZ / "data" / "online_retail_II.parquet"
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Códigos que no son productos: portes, comisiones, ajustes, muestras, vales...
NO_PRODUCTO = {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "ADJUST", "ADJUST2",
               "AMAZONFEE", "PADS", "CRUK", "TEST001", "TEST002", "SP1002"}


def cargar_crudo() -> pd.DataFrame:
    """Lee las dos hojas del Excel (descargándolo si hace falta) y cachea en parquet."""
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    if not XLSX.exists():
        XLSX.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=300) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                XLSX.write_bytes(z.read("online_retail_II.xlsx"))
    hojas = pd.read_excel(XLSX, sheet_name=None, engine="calamine")
    df = pd.concat(hojas.values(), ignore_index=True)
    for c in ["Invoice", "StockCode", "Description"]:
        df[c] = df[c].astype(str)
    df.to_parquet(PARQUET, index=False)
    return df


def limpiar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (ventas, devoluciones, informe).

    - ventas: líneas de compra válidas de clientes identificados.
    - devoluciones: líneas de facturas canceladas (C...), en positivo.
    - informe: cuántas filas quita cada regla, en orden.
    """
    df = crudo.rename(columns={
        "Invoice": "factura", "StockCode": "producto", "Description": "descripcion",
        "Quantity": "cantidad", "InvoiceDate": "fecha", "Price": "precio",
        "Customer ID": "cliente", "Country": "pais",
    })
    pasos = [("Filas originales", len(df))]

    df = df.drop_duplicates()
    pasos.append(("Sin duplicados (las dos hojas se solapan en dic. 2010)", len(df)))

    df = df[df["cliente"].notna()].copy()
    df["cliente"] = df["cliente"].astype(int)
    pasos.append(("Con cliente identificado", len(df)))

    codigo = df["producto"].str.upper()
    df = df[~codigo.isin(NO_PRODUCTO) & ~codigo.str.startswith("GIFT_")]
    pasos.append(("Solo productos (sin portes, comisiones ni vales)", len(df)))

    cancelada = df["factura"].str.startswith("C")
    devoluciones = df[cancelada].copy()
    devoluciones["cantidad"] = devoluciones["cantidad"].abs()
    devoluciones["importe"] = devoluciones["cantidad"] * devoluciones["precio"]

    ventas = df[~cancelada & (df["cantidad"] > 0) & (df["precio"] > 0)].copy()
    ventas["importe"] = ventas["cantidad"] * ventas["precio"]
    pasos.append(("Ventas válidas (sin devoluciones, cantidades ni precios <= 0)", len(ventas)))

    informe = pd.DataFrame(pasos, columns=["paso", "filas"])
    informe["filas_quitadas"] = (-informe["filas"].diff()).fillna(0).astype(int)
    return ventas.reset_index(drop=True), devoluciones.reset_index(drop=True), informe


def transacciones(ventas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cliente y día de compra (varias facturas el mismo día cuentan como una)."""
    t = (ventas.assign(dia=ventas["fecha"].dt.normalize())
         .groupby(["cliente", "dia"], as_index=False)["importe"].sum())
    return t
