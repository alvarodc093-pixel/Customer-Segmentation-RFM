"""Limpieza, tabla RFM y segmentos sobre datos hechos a mano."""

import pandas as pd
import pytest

from rfm.limpieza import limpiar, transacciones
from rfm.segmentos import ACCIONES, SEGMENTOS, puntuar, resumen_segmentos, segmento, tabla_rfm

CORTE = pd.Timestamp("2011-03-01")


@pytest.fixture
def crudo():
    filas = [
        # factura, producto, cantidad, fecha, precio, cliente
        ("1001", "85048", 2, "2011-01-10 10:00", 5.0, 1.0),
        ("1001", "85048", 2, "2011-01-10 10:00", 5.0, 1.0),      # duplicado exacto
        ("1001", "POST", 1, "2011-01-10 10:00", 18.0, 1.0),      # portes: fuera
        ("1002", "22041", 1, "2011-02-20 12:00", 30.0, 1.0),
        ("1003", "22041", 4, "2011-02-25 09:00", 10.0, 2.0),
        ("1004", "22041", 1, "2011-02-26 09:00", 10.0, None),    # sin cliente: fuera
        ("C1005", "22041", -1, "2011-02-27 09:00", 10.0, 2.0),   # devolución
        ("1006", "22041", 3, "2011-02-27 09:00", 0.0, 2.0),      # precio 0: fuera
        ("1007", "gift_0001_20", 1, "2011-02-27 09:00", 20.0, 2.0),  # vale: fuera
    ]
    return pd.DataFrame(filas, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]).assign(
        InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]), Description="x", Country="United Kingdom")


def test_limpieza(crudo):
    ventas, dev, informe = limpiar(crudo)
    assert sorted(ventas["factura"]) == ["1001", "1002", "1003"]
    assert ventas["importe"].sum() == pytest.approx(10 + 30 + 40)
    assert len(dev) == 1 and dev["importe"].iloc[0] == pytest.approx(10)
    assert informe["filas"].iloc[0] == len(crudo)
    assert informe["filas_quitadas"].sum() == len(crudo) - len(ventas)


def test_transacciones_agrupan_por_dia(crudo):
    ventas, _, _ = limpiar(crudo)
    extra = ventas.iloc[[0]].assign(factura="1008")       # otra factura el mismo día
    t = transacciones(pd.concat([ventas, extra]))
    assert len(t[t["cliente"] == 1]) == 2


def test_tabla_rfm(crudo):
    ventas, dev, _ = limpiar(crudo)
    rfm = tabla_rfm(ventas, dev, CORTE)
    assert rfm.loc[1, "frecuencia"] == 2
    assert rfm.loc[1, "recencia"] == 9                   # 20-feb -> 1-mar
    assert rfm.loc[2, "importe"] == pytest.approx(40 - 10)  # neto de devoluciones


def test_puntuar_reparte_en_quintiles():
    n = 100
    rfm = pd.DataFrame({"recencia": range(n), "frecuencia": [1] * 60 + list(range(2, 42)),
                        "importe": range(n)})
    r = puntuar(rfm)
    for c in "RFM":
        assert r[c].value_counts().tolist() == [20] * 5
    # recencia baja (compra reciente) = puntuación alta
    assert r.loc[r["recencia"].idxmin(), "R"] == 5


def test_todas_las_combinaciones_tienen_segmento():
    nombres = {segmento(r, f) for r in range(1, 6) for f in range(1, 6)}
    assert nombres == {n for n, _ in SEGMENTOS}
    assert set(ACCIONES) == nombres
    assert segmento(5, 5) == "Campeones" and segmento(1, 1) == "Hibernando"
    assert segmento(1, 5) == "No se pueden perder" and segmento(5, 1) == "Nuevos"


def test_resumen_segmentos_suma_100():
    rfm = pd.DataFrame({"recencia": range(50), "frecuencia": range(1, 51), "importe": range(50)})
    s = resumen_segmentos(puntuar(rfm))
    assert s["pct_clientes"].sum() == pytest.approx(1)
    assert s["clientes"].sum() == 50
