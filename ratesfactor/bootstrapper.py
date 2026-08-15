import math

import numpy as np
import pandas as pd

from .pricing import year_fraction


def log_interpolator(t, d1, d2, t1, t2):
    log_d1 = math.log(d1)
    log_d2 = math.log(d2)
    log_d = log_d1 + ((log_d2 - log_d1) / (t2 - t1)) * (t - t1)
    return math.exp(log_d)


def DF(t, discount_factors):
    t_vals = sorted(discount_factors.keys())
    if not t_vals:
        raise ValueError("No discount factors are available for interpolation.")

    for t_known in t_vals:
        if np.isclose(t, t_known):
            return discount_factors[t_known]

    if t < t_vals[0]:
        return discount_factors[t_vals[0]]
    if t > t_vals[-1]:
        return discount_factors[t_vals[-1]]

    for idx in range(len(t_vals) - 1):
        t1 = t_vals[idx]
        t2 = t_vals[idx + 1]
        if t1 < t < t2:
            return log_interpolator(t, discount_factors[t1], discount_factors[t2], t1, t2)

    raise ValueError(f"Could not interpolate discount factor for t={t}")


def _bootstrap_regular_tenor_row(par_curve, time_to_maturities, frequency):
    df = {}
    r = {}
    price = 100.0
    face_value = 100.0
    coupon_period = 1 / frequency

    for T in time_to_maturities:
        c = float(par_curve[T]) / 100.0

        if T <= coupon_period:
            df_t = pow(1 + c / frequency, -frequency * T)
        else:
            npv_cf = 0.0
            final_cf = face_value * (1 + c / frequency)
            pay_t = T - coupon_period

            while pay_t > 0:
                cf = face_value * c / frequency
                npv_cf += cf * DF(pay_t, df)
                pay_t -= coupon_period

            df_t = (price - npv_cf) / final_cf

        if df_t <= 0:
            raise ValueError(f"Bootstrapped non-positive discount factor for tenor {T}")

        df[T] = df_t
        r[T] = frequency * (pow(df_t, -1 / (frequency * T)) - 1)

    return r


def _DF_vector(t, discount_factors):
    t_vals = sorted(discount_factors.keys())
    if not t_vals:
        raise ValueError("No discount factors are available for interpolation.")

    for t_known in t_vals:
        if np.isclose(t, t_known):
            return discount_factors[t_known]

    if t < t_vals[0]:
        return discount_factors[t_vals[0]]
    if t > t_vals[-1]:
        return discount_factors[t_vals[-1]]

    for idx in range(len(t_vals) - 1):
        t1 = t_vals[idx]
        t2 = t_vals[idx + 1]
        if t1 < t < t2:
            log_d1 = np.log(discount_factors[t1])
            log_d2 = np.log(discount_factors[t2])
            log_d = log_d1 + ((log_d2 - log_d1) / (t2 - t1)) * (t - t1)
            return np.exp(log_d)

    raise ValueError(f"Could not interpolate discount factor for t={t}")


def _bootstrap_regular_tenor_history(par_yields, time_to_maturities, frequency):
    discount_factors = {}
    zero_rates = {}
    price = 100.0
    face_value = 100.0
    coupon_period = 1 / frequency

    for T in time_to_maturities:
        coupon = par_yields[T].to_numpy(dtype=float) / 100.0

        if T <= coupon_period:
            df_t = np.power(1 + coupon / frequency, -frequency * T)
        else:
            npv_cf = np.zeros(len(par_yields), dtype=float)
            final_cf = face_value * (1 + coupon / frequency)
            pay_t = T - coupon_period

            while pay_t > 1e-12:
                cf = face_value * coupon / frequency
                npv_cf += cf * _DF_vector(pay_t, discount_factors)
                pay_t -= coupon_period

            df_t = (price - npv_cf) / final_cf

        if np.any(df_t <= 0):
            raise ValueError(f"Bootstrapped non-positive discount factor for tenor {T}")

        discount_factors[T] = df_t
        zero_rates[T] = frequency * (np.power(df_t, -1 / (frequency * T)) - 1)

    return pd.DataFrame(zero_rates, index=par_yields.index) * 100.0


def bootstrap_zero_from_par(par_yields, frequency=2, dc_conv="ACT/ACT", use_calendar_dates=False):
    """Bootstrap proxy periodic zero rates from FRED/CMT par-yield history.

    This follows the research-notebook logic: each CMT tenor is treated as a
    par instrument priced at 100, shorter-than-coupon-period tenors are treated
    as deposit-style zero-coupon points, and coupon tenors are solved in
    maturity order using log-discount-factor interpolation for prior cashflows.

    By default the function uses the regular tenor grid directly, which is much
    faster for FRED CMT history because those are synthetic par curve nodes
    rather than actual CUSIPs with instrument-specific coupon calendars.
    """
    if 12 % int(frequency) != 0:
        raise ValueError("frequency must divide 12")

    par_yields = par_yields.copy()
    par_yields.index = pd.to_datetime(par_yields.index).normalize()
    par_yields = par_yields.sort_index()
    time_to_maturities = [float(col) for col in par_yields.columns]
    par_yields.columns = time_to_maturities

    res = []
    frequency = int(frequency)
    price = 100.0
    face_value = 100.0
    cf_period_months = 12 // frequency

    if not use_calendar_dates:
        return _bootstrap_regular_tenor_history(par_yields, time_to_maturities, frequency)

    for curve_date in par_yields.index:
        df = {}
        r = {}
        start_date = pd.Timestamp(curve_date)

        for T in time_to_maturities:
            months_to_maturity = int(round(float(T) * 12))
            end_date = start_date + pd.DateOffset(months=months_to_maturity)
            T_yf = year_fraction(start_date, end_date, dc_conv)
            c = float(par_yields.loc[curve_date, T]) / 100.0

            if T_yf <= 0:
                raise ValueError(f"Non-positive maturity for tenor {T}")

            if months_to_maturity <= cf_period_months:
                df_t = pow(1 + c / frequency, -frequency * T_yf)
            else:
                npv_cf = 0.0
                final_cf = 0.0
                cashflow_date = end_date

                while cashflow_date > start_date:
                    yf = year_fraction(start_date, cashflow_date, dc_conv)

                    if cashflow_date == end_date:
                        final_cf = face_value * (1 + c / frequency)
                    else:
                        cf = face_value * c / frequency
                        npv_cf += cf * DF(yf, df)

                    cashflow_date -= pd.DateOffset(months=cf_period_months)

                df_t = (price - npv_cf) / final_cf

            if df_t <= 0:
                raise ValueError(f"Bootstrapped non-positive discount factor for tenor {T}")

            df[T_yf] = df_t
            r[T] = frequency * (pow(df_t, -1 / (frequency * T_yf)) - 1)

        res.append(r)

    zero_rates = pd.DataFrame(res, index=par_yields.index)
    zero_rates = zero_rates.loc[:, time_to_maturities]
    return zero_rates * 100.0
