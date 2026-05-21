import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


def compare_arima_models(
    y,
    p_values=range(0, 5),
    d_values=(0,),
    q_values=range(0, 5),
    max_ljungbox_lag=10,
    trend="c",
):
    """
    Fit multiple ARIMA(p,d,q) models and compare by AIC, BIC, HQIC,
    log-likelihood, and residual Ljung-Box p-value.

    Parameters
    ----------
    y               : pd.Series or array-like  — time series
    p_values        : iterable of AR orders
    d_values        : iterable of differencing orders
    q_values        : iterable of MA orders
    max_ljungbox_lag: lag used for the residual autocorrelation test
    trend           : 'c' for constant, 'n' for no constant

    Returns
    -------
    pd.DataFrame sorted by BIC (converged models only)
    """
    results = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = ARIMA(
                            y,
                            order=order,
                            trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()

                    resid     = fit.resid.dropna() if hasattr(fit.resid, "dropna") else fit.resid
                    lb        = acorr_ljungbox(resid, lags=[max_ljungbox_lag], return_df=True)
                    lb_pvalue = lb["lb_pvalue"].iloc[0]

                    results.append({
                        "order":      order,
                        "p": p, "d": d, "q": q,
                        "aic":        fit.aic,
                        "bic":        fit.bic,
                        "hqic":       fit.hqic,
                        "loglik":     fit.llf,
                        "ljungbox_p": lb_pvalue,
                        "converged":  fit.mle_retvals.get("converged", None),
                    })

                except Exception as e:
                    results.append({
                        "order":      order,
                        "p": p, "d": d, "q": q,
                        "aic": None, "bic": None, "hqic": None,
                        "loglik": None, "ljungbox_p": None,
                        "converged": False,
                        "error":      str(e),
                    })

    table = pd.DataFrame(results)
    return table.dropna(subset=["aic", "bic"]).sort_values("bic").reset_index(drop=True)
