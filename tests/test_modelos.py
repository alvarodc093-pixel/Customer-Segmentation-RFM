"""Los modelos recuperan parámetros conocidos y predicen bien sobre datos simulados."""

import numpy as np
import pandas as pd
import pytest

from rfm.clv import BGNBD, GammaGamma, clv, resumen_cliente, simular_bgnbd

VERDAD = dict(r=0.25, alpha=4.0, a=0.8, b=2.5)


@pytest.fixture(scope="module")
def simulados():
    return simular_bgnbd(6000, **VERDAD, T=52, semilla=1, t_futuro=26)


@pytest.fixture(scope="module")
def bg(simulados):
    return BGNBD().ajustar(simulados["x"], simulados["t_x"], simulados["T"])


def test_bgnbd_recupera_parametros(bg):
    assert bg.r == pytest.approx(VERDAD["r"], rel=0.15)
    assert bg.a == pytest.approx(VERDAD["a"], rel=0.25)
    assert bg.b == pytest.approx(VERDAD["b"], rel=0.25)
    assert bg.alpha == pytest.approx(VERDAD["alpha"], rel=0.25)


def test_bgnbd_predice_compras_futuras(bg, simulados):
    pred = bg.compras_esperadas(26, simulados["x"], simulados["t_x"], simulados["T"])
    assert pred.sum() == pytest.approx(simulados["futuras"].sum(), rel=0.05)
    # también por grupos de frecuencia, no solo en total
    for k in range(4):
        m = (simulados["x"] == k).to_numpy()
        assert pred[m].mean() == pytest.approx(simulados["futuras"][m].mean(), abs=0.05)


def test_verosimilitud_maxima_en_el_ajuste(bg, simulados):
    x, tx, T = simulados["x"].to_numpy(), simulados["t_x"].to_numpy(), simulados["T"].to_numpy()
    ll = lambda p: BGNBD._loglik(np.log(p), x, tx, T).sum()  # noqa: E731
    ajustado = ll([bg.r, bg.alpha, bg.a, bg.b])
    assert ajustado >= ll(list(VERDAD.values())) - 1e-6
    assert ajustado > ll([bg.r * 1.2, bg.alpha, bg.a, bg.b])


def test_prob_activo():
    bg = BGNBD(**VERDAD)
    # sin compras repetidas, el modelo asume que sigue activo
    assert bg.prob_activo(0, 0, 30) == pytest.approx(1)
    # mismo número de compras: quien compró hace poco está más vivo
    assert bg.prob_activo(5, 29, 30) > bg.prob_activo(5, 5, 30)


def test_compras_esperadas_nuevo_coincide_con_historial_vacio():
    bg = BGNBD(**VERDAD)
    # un cliente con x=0 y T≈0 es como uno recién llegado
    assert bg.compras_esperadas(26, 0, 0, 1e-9) == pytest.approx(bg.compras_esperadas_nuevo(26), rel=1e-6)


def test_gamma_gamma_recupera_parametros():
    rng = np.random.default_rng(3)
    p, q, v = 6.0, 4.0, 15.0
    x = rng.integers(1, 20, 5000)
    nu = rng.gamma(q, 1 / v, 5000)
    m = np.array([rng.gamma(p * xi, 1 / (ni * xi)) for xi, ni in zip(x, nu)])
    gg = GammaGamma().ajustar(x, m)
    assert gg.p == pytest.approx(p, rel=0.2)
    assert gg.q == pytest.approx(q, rel=0.2)
    assert gg.v == pytest.approx(v, rel=0.2)
    # el importe esperado queda entre la media del cliente y la de la población
    e = gg.importe_esperado([1, 50], [10.0, 10.0])
    poblacion = gg.p * gg.v / (gg.q - 1)
    assert min(10, poblacion) < e[0] < max(10, poblacion)
    assert abs(e[1] - 10) < abs(e[0] - 10)


def test_resumen_cliente():
    t = pd.DataFrame({
        "cliente": [1, 1, 1, 2],
        "dia": pd.to_datetime(["2011-01-03", "2011-01-17", "2011-02-14", "2011-02-07"]),
        "importe": [100.0, 50.0, 70.0, 30.0],
    })
    r = resumen_cliente(t, pd.Timestamp("2011-02-28"))
    assert r.loc[1, "x"] == 2
    assert r.loc[1, "t_x"] == pytest.approx(6)        # 3-ene -> 14-feb = 6 semanas
    assert r.loc[1, "T"] == pytest.approx(8)          # 3-ene -> 28-feb = 8 semanas
    assert r.loc[1, "m"] == pytest.approx(60)         # (50 + 70) / 2, sin la primera compra
    assert r.loc[2, "x"] == 0 and r.loc[2, "m"] == 0


def test_clv_es_compras_por_importe():
    bg, gg = BGNBD(**VERDAD), GammaGamma(6, 4, 15)
    d = pd.DataFrame({"x": [3], "t_x": [20.0], "T": [30.0], "m": [12.0]})
    out = clv(bg, gg, d, 52, margen=0.3)
    assert out["clv"].iloc[0] == pytest.approx(
        out["compras_esperadas"].iloc[0] * out["importe_esperado"].iloc[0] * 0.3)
