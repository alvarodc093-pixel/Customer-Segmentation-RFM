"""Valor de vida del cliente (CLV) con BG/NBD + Gamma-Gamma, implementados con scipy.

- BG/NBD (Fader, Hardie y Lee, 2005): cuántas compras hará cada cliente,
  teniendo en cuenta que puede haberse ido sin avisar.
- Gamma-Gamma (Fader, Hardie y Lee, 2005b): cuánto gastará de media por compra.

Datos de entrada por cliente (unidad de tiempo: semanas):
    x   = compras repetidas (todas menos la primera)
    t_x = momento de la última compra, contado desde la primera
    T   = antigüedad: desde la primera compra hasta la fecha de corte
    m   = importe medio de las compras repetidas
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special


def resumen_cliente(trans: pd.DataFrame, fecha_corte: pd.Timestamp, semanas: bool = True) -> pd.DataFrame:
    """Pasa transacciones (cliente, dia, importe) al formato x, t_x, T, m."""
    t = trans[trans["dia"] < fecha_corte]
    unidad = 7.0 if semanas else 1.0
    g = t.groupby("cliente")
    primera = g["dia"].min()
    ultima = g["dia"].max()
    n = g.size()
    total = g["importe"].sum()
    # importe medio de las compras repetidas = (total - primera compra) / repetidas
    importe_primera = t.sort_values("dia").groupby("cliente")["importe"].first()
    x = n - 1
    m = ((total - importe_primera) / x.replace(0, np.nan)).fillna(0)
    return pd.DataFrame({
        "x": x,
        "t_x": (ultima - primera).dt.days / unidad,
        "T": (fecha_corte.normalize() - primera).dt.days / unidad,
        "m": m,
    })


def _minimizar(objetivo, x0):
    """Optimiza en escala logarítmica (parámetros siempre positivos) con límites."""
    def seguro(p):
        valor = objetivo(p)
        return valor if np.isfinite(valor) else 1e300
    res = optimize.minimize(seguro, x0=x0, method="L-BFGS-B", bounds=[(-12, 12)] * len(x0))
    if not res.success:
        res = optimize.minimize(seguro, x0=res.x, method="Nelder-Mead",
                                options={"maxiter": 20_000, "xatol": 1e-9, "fatol": 1e-9})
    return res


# ------------------------------------------------------------------ BG/NBD --

@dataclass
class BGNBD:
    r: float = np.nan
    alpha: float = np.nan
    a: float = np.nan
    b: float = np.nan

    @staticmethod
    def _loglik(params, x, t_x, T):
        r, alpha, a, b = np.exp(params)
        a1 = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
        a2 = special.gammaln(a + b) + special.gammaln(b + x) - special.gammaln(b) - special.gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
        return a1 + a2 + np.logaddexp(a3, a4)

    def ajustar(self, x, t_x, T) -> "BGNBD":
        x, t_x, T = map(np.asarray, (x, t_x, T))
        res = _minimizar(lambda p: -self._loglik(p, x, t_x, T).sum(), np.log([0.5, 5.0, 0.8, 2.0]))
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        self.loglik_ = -res.fun
        return self

    def prob_activo(self, x, t_x, T):
        x, t_x, T = map(np.asarray, (x, t_x, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return 1.0 / (1.0 + d)

    def compras_esperadas(self, t, x, t_x, T):
        """Compras esperadas en las próximas t semanas, dado el historial de cada cliente."""
        x, t_x, T = map(np.asarray, (x, t_x, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = special.hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = 1 + np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return num / den

    def compras_esperadas_nuevo(self, t):
        """Compras esperadas en t semanas para un cliente recién llegado (sin historial)."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = special.hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** r * hyp)


# ------------------------------------------------------------- Gamma-Gamma --

@dataclass
class GammaGamma:
    p: float = np.nan
    q: float = np.nan
    v: float = np.nan

    @staticmethod
    def _loglik(params, x, m):
        p, q, v = np.exp(params)
        return (special.gammaln(p * x + q) - special.gammaln(p * x) - special.gammaln(q)
                + q * np.log(v) + (p * x - 1) * np.log(m) + (p * x) * np.log(x)
                - (p * x + q) * np.log(x * m + v))

    def ajustar(self, x, m) -> "GammaGamma":
        """Solo con clientes que repiten (x > 0) y con importe medio positivo."""
        x, m = np.asarray(x, float), np.asarray(m, float)
        ok = (x > 0) & (m > 0)
        x, m = x[ok], m[ok]
        res = _minimizar(lambda pr: -self._loglik(pr, x, m).sum(), np.log([1.0, 1.0, 1.0]))
        self.p, self.q, self.v = np.exp(res.x)
        return self

    def importe_esperado(self, x, m):
        """Importe medio esperado por compra. Sin compras repetidas, la media de la población."""
        x, m = np.asarray(x, float), np.asarray(m, float)
        p, q, v = self.p, self.q, self.v
        media_poblacion = p * v / (q - 1)
        return np.where(x > 0, (p * (v + x * m)) / (p * x + q - 1), media_poblacion)


def clv(bg: BGNBD, gg: GammaGamma, datos: pd.DataFrame, semanas: float, margen: float = 1.0) -> pd.DataFrame:
    """Compras esperadas, importe esperado y CLV (sin descontar) para un horizonte dado."""
    out = datos.copy()
    out["prob_activo"] = bg.prob_activo(datos["x"], datos["t_x"], datos["T"])
    out["compras_esperadas"] = bg.compras_esperadas(semanas, datos["x"], datos["t_x"], datos["T"])
    out["importe_esperado"] = gg.importe_esperado(datos["x"], datos["m"])
    out["clv"] = out["compras_esperadas"] * out["importe_esperado"] * margen
    return out


# ---------------------------------------------------------------- simular --

def simular_bgnbd(n: int, r: float, alpha: float, a: float, b: float, T: float,
                  semilla: int = 0, t_futuro: float = 0.0) -> pd.DataFrame:
    """Simula clientes del proceso BG/NBD (para tests): devuelve x, t_x, T y compras futuras."""
    rng = np.random.default_rng(semilla)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    filas = []
    for i in range(n):
        t, compras, vivo = 0.0, [], True
        while vivo:
            t += rng.exponential(1 / lam[i])
            if t > T + t_futuro:
                break
            compras.append(t)
            vivo = rng.random() > p[i]
        pasado = [c for c in compras if c <= T]
        futuro = [c for c in compras if T < c <= T + t_futuro]
        filas.append((len(pasado), pasado[-1] if pasado else 0.0, T, len(futuro)))
    # En BG/NBD el tiempo empieza en la primera compra: x son compras tras esa.
    # Aquí el cliente "nace" en 0 con una compra implícita, así que x = compras en (0, T].
    return pd.DataFrame(filas, columns=["x", "t_x", "T", "futuras"])
