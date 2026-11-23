from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =============================================================================
# Model functions
# =============================================================================

def gaussian_component(freq, amp, mu, sigma):
    """
    Gaussian peak in frequency space
    """

    freq = np.asarray(freq, dtype=float)

    return amp * np.exp(-0.5 * ((freq - mu) / sigma) ** 2)


def model_background_only(freq, slope, intercept):
    """
    1/f background in dB space
    """

    freq = np.asarray(freq, dtype=float)

    return slope * np.log10(freq) + intercept


def model_background_alpha(
    freq,
    slope,
    intercept,
    alpha_amp,
    alpha_mu,
    alpha_sigma,
):
    """
    1/f background plus one alpha Gaussian peak
    """

    background = model_background_only(
        freq=freq,
        slope=slope,
        intercept=intercept,
    )

    alpha_peak = gaussian_component(
        freq=freq,
        amp=alpha_amp,
        mu=alpha_mu,
        sigma=alpha_sigma,
    )

    return background + alpha_peak


def model_background_beta(
    freq,
    slope,
    intercept,
    beta_amp,
    beta_mu,
    beta_sigma,
):
    """
    1/f background plus one beta Gaussian peak
    """

    background = model_background_only(
        freq=freq,
        slope=slope,
        intercept=intercept,
    )

    beta_peak = gaussian_component(
        freq=freq,
        amp=beta_amp,
        mu=beta_mu,
        sigma=beta_sigma,
    )

    return background + beta_peak


def model_background_alpha_beta(
    freq,
    slope,
    intercept,
    alpha_amp,
    alpha_mu,
    alpha_sigma,
    beta_amp,
    beta_mu,
    beta_sigma,
):
    """
    1/f background plus alpha and beta Gaussian peaks
    """

    background = model_background_only(
        freq=freq,
        slope=slope,
        intercept=intercept,
    )

    alpha_peak = gaussian_component(
        freq=freq,
        amp=alpha_amp,
        mu=alpha_mu,
        sigma=alpha_sigma,
    )

    beta_peak = gaussian_component(
        freq=freq,
        amp=beta_amp,
        mu=beta_mu,
        sigma=beta_sigma,
    )

    return background + alpha_peak + beta_peak


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_psd_db_over_batch(
    input_csv,
    value_col="psd_db",
    batch_col="batch",
    condition_col="condition",
    channel_col="channel",
    freq_col="frequency_hz",
    ddof=1,
):
    """
    Aggregate PSD dB values over batch for each condition, channel, and frequency bin
    """

    input_path = Path(input_csv)
    df = pd.read_csv(input_path)

    required_cols = [
        batch_col,
        condition_col,
        channel_col,
        freq_col,
        value_col,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {input_path}: {missing_cols}"
        )

    group_cols = [
        condition_col,
        channel_col,
        freq_col,
    ]

    # Average duplicate rows within each batch first
    batch_level = (
        df
        .groupby(group_cols + [batch_col], as_index=False)[value_col]
        .mean()
    )

    aggregated = (
        batch_level
        .groupby(group_cols)[value_col]
        .agg(
            mean_psd_db="mean",
            std_psd_db=lambda x: x.std(ddof=ddof),
            n_batches="count",
        )
        .reset_index()
    )

    aggregated["sem_psd_db"] = (
        aggregated["std_psd_db"] / np.sqrt(aggregated["n_batches"])
    )

    aggregated = (
        aggregated
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    return aggregated


# =============================================================================
# Fit helpers
# =============================================================================

def get_initial_background_fit(
    freq_fit,
    psd_fit,
    alpha_mu_bounds=(7.5, 13.0),
    beta_mu_bounds=(14.0, 30.0),
):
    """
    Estimate initial 1/f background while excluding alpha and beta bands
    """

    log_freq_fit = np.log10(freq_fit)

    exclude_alpha = (
        (freq_fit >= alpha_mu_bounds[0])
        & (freq_fit <= alpha_mu_bounds[1])
    )

    exclude_beta = (
        (freq_fit >= beta_mu_bounds[0])
        & (freq_fit <= beta_mu_bounds[1])
    )

    background_mask = ~(exclude_alpha | exclude_beta)

    if background_mask.sum() >= 2:
        slope, intercept = np.polyfit(
            log_freq_fit[background_mask],
            psd_fit[background_mask],
            deg=1,
        )
    else:
        slope, intercept = np.polyfit(
            log_freq_fit,
            psd_fit,
            deg=1,
        )

    return slope, intercept


def get_peak_initial_guess(
    freq_fit,
    residual_fit,
    mu_guess,
    mu_bounds,
    amp_bounds,
    sigma_guess,
    sigma_bounds,
):
    """
    Estimate initial Gaussian parameters from residuals inside a target band
    """

    band_mask = (
        (freq_fit >= mu_bounds[0])
        & (freq_fit <= mu_bounds[1])
    )

    if band_mask.any():
        band_freq = freq_fit[band_mask]
        band_residual = residual_fit[band_mask]

        best_idx = np.nanargmax(band_residual)

        init_amp = band_residual[best_idx]
        init_mu = band_freq[best_idx]
    else:
        init_amp = 1.0
        init_mu = mu_guess

    init_amp = np.clip(
        init_amp,
        amp_bounds[0] + 1e-6,
        amp_bounds[1],
    )

    init_mu = np.clip(
        init_mu,
        mu_bounds[0],
        mu_bounds[1],
    )

    init_sigma = np.clip(
        sigma_guess,
        sigma_bounds[0],
        sigma_bounds[1],
    )

    return init_amp, init_mu, init_sigma


def get_sem_weights(
    group_df,
    valid_mask,
    sem_col="sem_psd_db",
    use_sem_weights=True,
):
    """
    Return SEM values for curve_fit when all values are finite and positive
    """

    if not use_sem_weights:
        return None

    if sem_col not in group_df.columns:
        return None

    sem = group_df.loc[valid_mask, sem_col].to_numpy(dtype=float)
    sem_is_valid = np.isfinite(sem) & (sem > 0)

    if sem_is_valid.all():
        return sem

    return None


def compute_fit_metrics(y_true, y_pred, n_parameters):
    """
    Compute RSS, MSE, RMSE, AIC, and AICc
    """

    n_points = len(y_true)
    residual = y_true - y_pred

    rss = float(np.sum(residual ** 2))
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))

    eps = np.finfo(float).eps
    rss_safe = max(rss, eps)

    aic = float(
        n_points * np.log(rss_safe / n_points)
        + 2 * n_parameters
    )

    if n_points > n_parameters + 1:
        aicc = float(
            aic
            + (2 * n_parameters * (n_parameters + 1))
            / (n_points - n_parameters - 1)
        )
    else:
        aicc = np.nan

    return {
        "rss": rss,
        "mse": mse,
        "rmse": rmse,
        "aic": aic,
        "aicc": aicc,
    }


def get_model_peak_count(model_name):
    """
    Return number of Gaussian peaks in model
    """

    peak_counts = {
        "background_only": 0,
        "background_alpha": 1,
        "background_beta": 1,
        "background_alpha_beta": 2,
    }

    return peak_counts[model_name]


def empty_fit_result(model_name, n_fit_points, n_parameters):
    """
    Create an empty fit result row
    """

    return {
        "model_name": model_name,
        "fit_status": "not_fit",
        "n_fit_points": n_fit_points,
        "n_parameters": n_parameters,
        "n_gaussian_peaks": get_model_peak_count(model_name),
        "rss": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "aic": np.nan,
        "aicc": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "alpha_amp_db": np.nan,
        "alpha_mu_hz": np.nan,
        "alpha_sigma_hz": np.nan,
        "beta_amp_db": np.nan,
        "beta_mu_hz": np.nan,
        "beta_sigma_hz": np.nan,
    }


def unpack_fit_parameters(model_name, popt, result):
    """
    Store fitted parameters into the result row
    """

    if model_name == "background_only":
        slope, intercept = popt

        result["slope"] = slope
        result["intercept"] = intercept

    elif model_name == "background_alpha":
        (
            slope,
            intercept,
            alpha_amp,
            alpha_mu,
            alpha_sigma,
        ) = popt

        result["slope"] = slope
        result["intercept"] = intercept
        result["alpha_amp_db"] = alpha_amp
        result["alpha_mu_hz"] = alpha_mu
        result["alpha_sigma_hz"] = alpha_sigma

    elif model_name == "background_beta":
        (
            slope,
            intercept,
            beta_amp,
            beta_mu,
            beta_sigma,
        ) = popt

        result["slope"] = slope
        result["intercept"] = intercept
        result["beta_amp_db"] = beta_amp
        result["beta_mu_hz"] = beta_mu
        result["beta_sigma_hz"] = beta_sigma

    elif model_name == "background_alpha_beta":
        (
            slope,
            intercept,
            alpha_amp,
            alpha_mu,
            alpha_sigma,
            beta_amp,
            beta_mu,
            beta_sigma,
        ) = popt

        result["slope"] = slope
        result["intercept"] = intercept
        result["alpha_amp_db"] = alpha_amp
        result["alpha_mu_hz"] = alpha_mu
        result["alpha_sigma_hz"] = alpha_sigma
        result["beta_amp_db"] = beta_amp
        result["beta_mu_hz"] = beta_mu
        result["beta_sigma_hz"] = beta_sigma

    return result


def fit_candidate_model(
    model_name,
    model_func,
    freq_fit,
    psd_fit,
    p0,
    lower_bounds,
    upper_bounds,
    sigma_y=None,
    maxfev=50000,
):
    """
    Fit one candidate model and return parameters plus fit metrics
    """

    n_parameters = len(p0)
    n_fit_points = len(freq_fit)

    result = empty_fit_result(
        model_name=model_name,
        n_fit_points=n_fit_points,
        n_parameters=n_parameters,
    )

    try:
        popt, _ = curve_fit(
            model_func,
            freq_fit,
            psd_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            sigma=sigma_y,
            absolute_sigma=False,
            maxfev=maxfev,
        )

        y_pred = model_func(freq_fit, *popt)

        result["fit_status"] = "success"
        result.update(
            compute_fit_metrics(
                y_true=psd_fit,
                y_pred=y_pred,
                n_parameters=n_parameters,
            )
        )

        result = unpack_fit_parameters(
            model_name=model_name,
            popt=popt,
            result=result,
        )

    except Exception as exc:
        result["fit_status"] = f"failed_{type(exc).__name__}"

    return result


# =============================================================================
# Peak validity and model selection
# =============================================================================

def check_peak_validity(
    row,
    band,
    mu_bounds,
    sigma_bounds,
    min_peak_amp_db=0.5,
    boundary_tol_hz=0.05,
):
    """
    Check whether a fitted Gaussian peak is usable for model selection
    """

    amp = row[f"{band}_amp_db"]
    mu = row[f"{band}_mu_hz"]
    sigma = row[f"{band}_sigma_hz"]

    reasons = []
    warnings = []

    if not np.isfinite(amp):
        reasons.append("missing_amp")

    if not np.isfinite(mu):
        reasons.append("missing_mu")

    if not np.isfinite(sigma):
        reasons.append("missing_sigma")

    if reasons:
        return False, ";".join(reasons), ""

    if amp < min_peak_amp_db:
        reasons.append("amp_too_small")

    if mu <= mu_bounds[0] + boundary_tol_hz:
        reasons.append("mu_at_lower_bound")

    if mu >= mu_bounds[1] - boundary_tol_hz:
        reasons.append("mu_at_upper_bound")

    if sigma < sigma_bounds[0]:
        reasons.append("sigma_below_lower_bound")

    if sigma > sigma_bounds[1]:
        reasons.append("sigma_above_upper_bound")

    if sigma <= sigma_bounds[0] + boundary_tol_hz:
        warnings.append("sigma_near_lower_bound")

    if sigma >= sigma_bounds[1] - boundary_tol_hz:
        warnings.append("sigma_near_upper_bound")

    is_valid = len(reasons) == 0

    return is_valid, ";".join(reasons), ";".join(warnings)


def add_selection_validity_columns(
    summary_df,
    alpha_mu_bounds=(7.5, 13.0),
    beta_mu_bounds=(14.0, 30.0),
    alpha_sigma_bounds=(0.5, 2.5),
    beta_sigma_bounds=(0.5, 4.0),
    min_peak_amp_db=0.5,
    boundary_tol_hz=0.05,
):
    """
    Add peak validity columns and candidate-selection validity
    """

    summary_df = summary_df.copy()

    alpha_valid_list = []
    alpha_reasons_list = []
    alpha_warnings_list = []

    beta_valid_list = []
    beta_reasons_list = []
    beta_warnings_list = []

    valid_for_selection_list = []

    for _, row in summary_df.iterrows():

        model_name = row["model_name"]

        alpha_valid = np.nan
        alpha_reasons = ""
        alpha_warnings = ""

        beta_valid = np.nan
        beta_reasons = ""
        beta_warnings = ""

        valid_for_selection = (
            row["fit_status"] == "success"
            and np.isfinite(row["aic"])
        )

        if model_name in ["background_alpha", "background_alpha_beta"]:
            (
                alpha_valid,
                alpha_reasons,
                alpha_warnings,
            ) = check_peak_validity(
                row=row,
                band="alpha",
                mu_bounds=alpha_mu_bounds,
                sigma_bounds=alpha_sigma_bounds,
                min_peak_amp_db=min_peak_amp_db,
                boundary_tol_hz=boundary_tol_hz,
            )

            valid_for_selection = valid_for_selection and alpha_valid

        if model_name in ["background_beta", "background_alpha_beta"]:
            (
                beta_valid,
                beta_reasons,
                beta_warnings,
            ) = check_peak_validity(
                row=row,
                band="beta",
                mu_bounds=beta_mu_bounds,
                sigma_bounds=beta_sigma_bounds,
                min_peak_amp_db=min_peak_amp_db,
                boundary_tol_hz=boundary_tol_hz,
            )

            valid_for_selection = valid_for_selection and beta_valid

        alpha_valid_list.append(alpha_valid)
        alpha_reasons_list.append(alpha_reasons)
        alpha_warnings_list.append(alpha_warnings)

        beta_valid_list.append(beta_valid)
        beta_reasons_list.append(beta_reasons)
        beta_warnings_list.append(beta_warnings)

        valid_for_selection_list.append(valid_for_selection)

    summary_df["alpha_peak_valid"] = alpha_valid_list
    summary_df["alpha_peak_reject_reason"] = alpha_reasons_list
    summary_df["alpha_peak_warning"] = alpha_warnings_list

    summary_df["beta_peak_valid"] = beta_valid_list
    summary_df["beta_peak_reject_reason"] = beta_reasons_list
    summary_df["beta_peak_warning"] = beta_warnings_list

    summary_df["valid_for_selection"] = valid_for_selection_list

    return summary_df


def select_model_with_aic_threshold(
    summary_df,
    min_delta_aic_for_extra_peak=3.0,
):
    """
    Select model using AIC with a minimum improvement threshold per extra peak
    """

    valid = summary_df[
        summary_df["valid_for_selection"]
        & np.isfinite(summary_df["aic"])
    ].copy()

    if valid.empty:
        raise RuntimeError("No valid candidate models available for selection")

    background_rows = valid[valid["model_name"] == "background_only"]

    if not background_rows.empty:
        selected = background_rows.sort_values("aic").iloc[0].copy()
        selection_reason = "background_only_start"
    else:
        selected = valid.sort_values("aic").iloc[0].copy()
        selection_reason = "lowest_valid_aic_no_background"

    for n_peaks in [1, 2]:

        candidate_rows = valid[valid["n_gaussian_peaks"] == n_peaks]

        if candidate_rows.empty:
            continue

        candidate = candidate_rows.sort_values("aic").iloc[0].copy()

        additional_peaks = (
            candidate["n_gaussian_peaks"]
            - selected["n_gaussian_peaks"]
        )

        if additional_peaks <= 0:
            continue

        aic_improvement = selected["aic"] - candidate["aic"]
        required_improvement = (
            min_delta_aic_for_extra_peak * additional_peaks
        )

        if aic_improvement >= required_improvement:
            selected = candidate
            selection_reason = (
                f"accepted_{candidate['model_name']}_"
                f"delta_aic={aic_improvement:.2f}_"
                f"required={required_improvement:.2f}"
            )

    selected["selection_reason"] = selection_reason

    return selected


def predict_selected_model_components(freq, fit_row):
    """
    Predict selected background, Gaussian components, and full selected model
    """

    freq = np.asarray(freq, dtype=float)

    selected_background = np.full_like(freq, np.nan, dtype=float)
    alpha = np.full_like(freq, np.nan, dtype=float)
    beta = np.full_like(freq, np.nan, dtype=float)
    full_model = np.full_like(freq, np.nan, dtype=float)

    slope = fit_row["slope"]
    intercept = fit_row["intercept"]

    positive_freq = freq > 0

    selected_background[positive_freq] = model_background_only(
        freq=freq[positive_freq],
        slope=slope,
        intercept=intercept,
    )

    full_model = selected_background.copy()

    alpha_amp = fit_row["alpha_amp_db"]
    alpha_mu = fit_row["alpha_mu_hz"]
    alpha_sigma = fit_row["alpha_sigma_hz"]

    beta_amp = fit_row["beta_amp_db"]
    beta_mu = fit_row["beta_mu_hz"]
    beta_sigma = fit_row["beta_sigma_hz"]

    if np.isfinite(alpha_amp):
        alpha = gaussian_component(
            freq=freq,
            amp=alpha_amp,
            mu=alpha_mu,
            sigma=alpha_sigma,
        )

        full_model = full_model + alpha

    if np.isfinite(beta_amp):
        beta = gaussian_component(
            freq=freq,
            amp=beta_amp,
            mu=beta_mu,
            sigma=beta_sigma,
        )

        full_model = full_model + beta

    return selected_background, alpha, beta, full_model


# =============================================================================
# Group-level model comparison
# =============================================================================

def fit_all_models_one_group(
    group_df,
    condition_value,
    channel_value,
    freq_col="frequency_hz",
    psd_col="mean_psd_db",
    sem_col="sem_psd_db",
    fit_freq_min=1.0,
    fit_freq_max=40.0,
    alpha_mu_guess=10.0,
    beta_mu_guess=20.0,
    alpha_mu_bounds=(7.5, 13.0),
    beta_mu_bounds=(14.0, 30.0),
    alpha_sigma_bounds=(0.5, 2.5),
    beta_sigma_bounds=(0.5, 4.0),
    amp_bounds=(0.0, 30.0),
    use_sem_weights=True,
    min_peak_amp_db=0.5,
    boundary_tol_hz=0.05,
    min_delta_aic_for_extra_peak=3.0,
):
    """
    Fit all candidate models for one condition and one channel
    """

    group_df = group_df.sort_values(freq_col).copy()

    freq = group_df[freq_col].to_numpy(dtype=float)
    psd = group_df[psd_col].to_numpy(dtype=float)

    valid = np.isfinite(freq) & np.isfinite(psd) & (freq > 0)

    if fit_freq_min is not None:
        valid &= freq >= fit_freq_min

    if fit_freq_max is not None:
        valid &= freq <= fit_freq_max

    if valid.sum() < 8:
        raise ValueError(
            f"Not enough valid frequency bins for condition={condition_value}, "
            f"channel={channel_value}"
        )

    freq_fit = freq[valid]
    psd_fit = psd[valid]

    sigma_y = get_sem_weights(
        group_df=group_df,
        valid_mask=valid,
        sem_col=sem_col,
        use_sem_weights=use_sem_weights,
    )

    init_slope, init_intercept = get_initial_background_fit(
        freq_fit=freq_fit,
        psd_fit=psd_fit,
        alpha_mu_bounds=alpha_mu_bounds,
        beta_mu_bounds=beta_mu_bounds,
    )

    init_background = model_background_only(
        freq=freq_fit,
        slope=init_slope,
        intercept=init_intercept,
    )

    init_residual = psd_fit - init_background

    init_alpha_amp, init_alpha_mu, init_alpha_sigma = get_peak_initial_guess(
        freq_fit=freq_fit,
        residual_fit=init_residual,
        mu_guess=alpha_mu_guess,
        mu_bounds=alpha_mu_bounds,
        amp_bounds=amp_bounds,
        sigma_guess=1.5,
        sigma_bounds=alpha_sigma_bounds,
    )

    init_beta_amp, init_beta_mu, init_beta_sigma = get_peak_initial_guess(
        freq_fit=freq_fit,
        residual_fit=init_residual,
        mu_guess=beta_mu_guess,
        mu_bounds=beta_mu_bounds,
        amp_bounds=amp_bounds,
        sigma_guess=2.0,
        sigma_bounds=beta_sigma_bounds,
    )

    candidate_specs = [
        {
            "model_name": "background_only",
            "model_func": model_background_only,
            "p0": [
                init_slope,
                init_intercept,
            ],
            "lower_bounds": [
                -np.inf,
                -np.inf,
            ],
            "upper_bounds": [
                np.inf,
                np.inf,
            ],
        },
        {
            "model_name": "background_alpha",
            "model_func": model_background_alpha,
            "p0": [
                init_slope,
                init_intercept,
                init_alpha_amp,
                init_alpha_mu,
                init_alpha_sigma,
            ],
            "lower_bounds": [
                -np.inf,
                -np.inf,
                amp_bounds[0],
                alpha_mu_bounds[0],
                alpha_sigma_bounds[0],
            ],
            "upper_bounds": [
                np.inf,
                np.inf,
                amp_bounds[1],
                alpha_mu_bounds[1],
                alpha_sigma_bounds[1],
            ],
        },
        {
            "model_name": "background_beta",
            "model_func": model_background_beta,
            "p0": [
                init_slope,
                init_intercept,
                init_beta_amp,
                init_beta_mu,
                init_beta_sigma,
            ],
            "lower_bounds": [
                -np.inf,
                -np.inf,
                amp_bounds[0],
                beta_mu_bounds[0],
                beta_sigma_bounds[0],
            ],
            "upper_bounds": [
                np.inf,
                np.inf,
                amp_bounds[1],
                beta_mu_bounds[1],
                beta_sigma_bounds[1],
            ],
        },
        {
            "model_name": "background_alpha_beta",
            "model_func": model_background_alpha_beta,
            "p0": [
                init_slope,
                init_intercept,
                init_alpha_amp,
                init_alpha_mu,
                init_alpha_sigma,
                init_beta_amp,
                init_beta_mu,
                init_beta_sigma,
            ],
            "lower_bounds": [
                -np.inf,
                -np.inf,
                amp_bounds[0],
                alpha_mu_bounds[0],
                alpha_sigma_bounds[0],
                amp_bounds[0],
                beta_mu_bounds[0],
                beta_sigma_bounds[0],
            ],
            "upper_bounds": [
                np.inf,
                np.inf,
                amp_bounds[1],
                alpha_mu_bounds[1],
                alpha_sigma_bounds[1],
                amp_bounds[1],
                beta_mu_bounds[1],
                beta_sigma_bounds[1],
            ],
        },
    ]

    summary_rows = []

    for spec in candidate_specs:
        result = fit_candidate_model(
            model_name=spec["model_name"],
            model_func=spec["model_func"],
            freq_fit=freq_fit,
            psd_fit=psd_fit,
            p0=spec["p0"],
            lower_bounds=spec["lower_bounds"],
            upper_bounds=spec["upper_bounds"],
            sigma_y=sigma_y,
        )

        result["condition"] = condition_value
        result["channel"] = channel_value

        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = add_selection_validity_columns(
        summary_df=summary_df,
        alpha_mu_bounds=alpha_mu_bounds,
        beta_mu_bounds=beta_mu_bounds,
        alpha_sigma_bounds=alpha_sigma_bounds,
        beta_sigma_bounds=beta_sigma_bounds,
        min_peak_amp_db=min_peak_amp_db,
        boundary_tol_hz=boundary_tol_hz,
    )

    best_fit = select_model_with_aic_threshold(
        summary_df=summary_df,
        min_delta_aic_for_extra_peak=min_delta_aic_for_extra_peak,
    )

    selected_background, alpha, beta, full_model = predict_selected_model_components(
        freq=freq,
        fit_row=best_fit,
    )

    fitted_group = group_df.copy()

    fitted_group["selected_model"] = best_fit["model_name"]
    fitted_group["selected_model_aic"] = best_fit["aic"]
    fitted_group["selected_model_aicc"] = best_fit["aicc"]
    fitted_group["selected_model_rss"] = best_fit["rss"]
    fitted_group["selected_model_mse"] = best_fit["mse"]
    fitted_group["selected_model_rmse"] = best_fit["rmse"]
    fitted_group["selected_model_reason"] = best_fit["selection_reason"]

    fitted_group["selected_background_1overf_db"] = selected_background
    fitted_group["background_1overf_db"] = selected_background

    fitted_group["alpha_gaussian_db"] = alpha
    fitted_group["beta_gaussian_db"] = beta
    fitted_group["best_model_fit_db"] = full_model

    fitted_group["psd_db_bg_removed"] = psd - selected_background
    fitted_group["psd_db_best_model_residual"] = psd - full_model

    fitted_group["one_over_f_slope"] = best_fit["slope"]
    fitted_group["one_over_f_intercept"] = best_fit["intercept"]

    fitted_group["alpha_amp_db"] = best_fit["alpha_amp_db"]
    fitted_group["alpha_mu_hz"] = best_fit["alpha_mu_hz"]
    fitted_group["alpha_sigma_hz"] = best_fit["alpha_sigma_hz"]

    fitted_group["beta_amp_db"] = best_fit["beta_amp_db"]
    fitted_group["beta_mu_hz"] = best_fit["beta_mu_hz"]
    fitted_group["beta_sigma_hz"] = best_fit["beta_sigma_hz"]

    fitted_group["fit_freq_min"] = fit_freq_min
    fitted_group["fit_freq_max"] = fit_freq_max

    summary_df["delta_aic_vs_selected"] = summary_df["aic"] - best_fit["aic"]
    summary_df["delta_aicc_vs_selected"] = summary_df["aicc"] - best_fit["aicc"]

    summary_df["is_selected_best_model"] = (
        summary_df["model_name"] == best_fit["model_name"]
    )

    summary_df["selection_reason"] = ""

    selected_mask = summary_df["is_selected_best_model"]
    summary_df.loc[selected_mask, "selection_reason"] = best_fit["selection_reason"]

    return fitted_group, summary_df


def add_best_model_fits_to_aggregated(
    aggregated_df,
    condition_col="condition",
    channel_col="channel",
    freq_col="frequency_hz",
    psd_col="mean_psd_db",
    sem_col="sem_psd_db",
    fit_freq_min=1.0,
    fit_freq_max=40.0,
    alpha_mu_guess=10.0,
    beta_mu_guess=20.0,
    alpha_mu_bounds=(7.5, 13.0),
    beta_mu_bounds=(14.0, 30.0),
    alpha_sigma_bounds=(0.5, 2.5),
    beta_sigma_bounds=(0.5, 4.0),
    amp_bounds=(0.0, 30.0),
    use_sem_weights=True,
    min_peak_amp_db=0.5,
    boundary_tol_hz=0.05,
    min_delta_aic_for_extra_peak=3.0,
):
    """
    Fit all candidate models separately for each condition and channel
    """

    fitted_groups = []
    summary_groups = []

    group_cols = [
        condition_col,
        channel_col,
    ]

    for (condition_value, channel_value), group_df in aggregated_df.groupby(
        group_cols,
        sort=False,
    ):

        fitted_group, summary_df = fit_all_models_one_group(
            group_df=group_df,
            condition_value=condition_value,
            channel_value=channel_value,
            freq_col=freq_col,
            psd_col=psd_col,
            sem_col=sem_col,
            fit_freq_min=fit_freq_min,
            fit_freq_max=fit_freq_max,
            alpha_mu_guess=alpha_mu_guess,
            beta_mu_guess=beta_mu_guess,
            alpha_mu_bounds=alpha_mu_bounds,
            beta_mu_bounds=beta_mu_bounds,
            alpha_sigma_bounds=alpha_sigma_bounds,
            beta_sigma_bounds=beta_sigma_bounds,
            amp_bounds=amp_bounds,
            use_sem_weights=use_sem_weights,
            min_peak_amp_db=min_peak_amp_db,
            boundary_tol_hz=boundary_tol_hz,
            min_delta_aic_for_extra_peak=min_delta_aic_for_extra_peak,
        )

        fitted_groups.append(fitted_group)
        summary_groups.append(summary_df)

    fitted_df = pd.concat(fitted_groups, ignore_index=True)
    summary_df = pd.concat(summary_groups, ignore_index=True)

    fitted_df = (
        fitted_df
        .sort_values([condition_col, channel_col, freq_col])
        .reset_index(drop=True)
    )

    summary_df = (
        summary_df
        .sort_values([condition_col, channel_col, "aic", "model_name"])
        .reset_index(drop=True)
    )

    return fitted_df, summary_df


# =============================================================================
# Peak summaries
# =============================================================================

def make_selected_gaussian_peak_summary(
    best_fit_df,
    condition_col="condition",
    channel_col="channel",
):
    """
    Create one row per selected Gaussian peak
    """

    unique_rows = (
        best_fit_df[
            [
                condition_col,
                channel_col,
                "selected_model",
                "selected_model_aic",
                "selected_model_reason",
                "alpha_amp_db",
                "alpha_mu_hz",
                "alpha_sigma_hz",
                "beta_amp_db",
                "beta_mu_hz",
                "beta_sigma_hz",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    peak_rows = []

    for _, row in unique_rows.iterrows():

        if np.isfinite(row["alpha_mu_hz"]):
            peak_rows.append(
                {
                    "condition": row[condition_col],
                    "channel": row[channel_col],
                    "selected_model": row["selected_model"],
                    "selected_model_aic": row["selected_model_aic"],
                    "selected_model_reason": row["selected_model_reason"],
                    "band": "alpha",
                    "amp_db": row["alpha_amp_db"],
                    "mean_mu_hz": row["alpha_mu_hz"],
                    "std_sigma_hz": row["alpha_sigma_hz"],
                }
            )

        if np.isfinite(row["beta_mu_hz"]):
            peak_rows.append(
                {
                    "condition": row[condition_col],
                    "channel": row[channel_col],
                    "selected_model": row["selected_model"],
                    "selected_model_aic": row["selected_model_aic"],
                    "selected_model_reason": row["selected_model_reason"],
                    "band": "beta",
                    "amp_db": row["beta_amp_db"],
                    "mean_mu_hz": row["beta_mu_hz"],
                    "std_sigma_hz": row["beta_sigma_hz"],
                }
            )

    return pd.DataFrame(peak_rows)


def summarize_selected_gaussian_peaks(peak_summary_df):
    """
    Summarize selected Gaussian peak parameters across condition and channel
    """

    if peak_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "band",
                "n_peaks",
                "mean_amp_db",
                "std_amp_db",
                "mean_mu_hz",
                "std_mu_hz",
                "mean_sigma_hz",
                "std_sigma_hz",
            ]
        )

    summary = (
        peak_summary_df
        .groupby("band")
        .agg(
            n_peaks=("band", "count"),
            mean_amp_db=("amp_db", "mean"),
            std_amp_db=("amp_db", "std"),
            mean_mu_hz=("mean_mu_hz", "mean"),
            std_mu_hz=("mean_mu_hz", "std"),
            mean_sigma_hz=("std_sigma_hz", "mean"),
            std_sigma_hz=("std_sigma_hz", "std"),
        )
        .reset_index()
    )

    return summary


# =============================================================================
# Plotting
# =============================================================================

def safe_filename(text):
    """
    Convert text to a safe filename fragment
    """

    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(":", "-")
    )


def clean_errorbar_values(yerr):
    """
    Convert invalid error bars to zero for safe plotting
    """

    yerr = np.asarray(yerr, dtype=float)

    return np.where(
        np.isfinite(yerr) & (yerr >= 0),
        yerr,
        0.0,
    )


def format_gaussian_peak_report(sub):
    """
    Format selected Gaussian peak parameters for plot annotation
    """

    selected_model = sub["selected_model"].iloc[0]
    selected_aic = sub["selected_model_aic"].iloc[0]
    selected_reason = sub["selected_model_reason"].iloc[0]

    alpha_amp = sub["alpha_amp_db"].iloc[0]
    alpha_mu = sub["alpha_mu_hz"].iloc[0]
    alpha_sigma = sub["alpha_sigma_hz"].iloc[0]

    beta_amp = sub["beta_amp_db"].iloc[0]
    beta_mu = sub["beta_mu_hz"].iloc[0]
    beta_sigma = sub["beta_sigma_hz"].iloc[0]

    lines = [
        f"Selected model: {selected_model}",
        f"AIC: {selected_aic:.2f}",
        f"Rule: {selected_reason}",
    ]

    '''
    if np.isfinite(alpha_mu):
        lines.append(
            "Alpha Gaussian: "
            f"μ={alpha_mu:.2f} Hz, "
            f"σ={alpha_sigma:.2f} Hz, "
            f"amp={alpha_amp:.2f} dB"
        )
    else:
        lines.append("Alpha Gaussian: not selected")

    if np.isfinite(beta_mu):
        lines.append(
            "Beta Gaussian: "
            f"μ={beta_mu:.2f} Hz, "
            f"σ={beta_sigma:.2f} Hz, "
            f"amp={beta_amp:.2f} dB"
        )
    else:
        lines.append("Beta Gaussian: not selected")
    '''
    return "\n".join(lines)


def plot_best_fit_for_channel(
    df,
    channel,
    output_dir,
    condition=None,
    freq_col="frequency_hz",
    condition_col="condition",
    channel_col="channel",
    mean_col="mean_psd_db",
    sem_col="sem_psd_db",
):
    """
    Save PSD plots using the selected AIC-best model for one channel
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_df = df[df[channel_col] == channel].copy()

    if channel_df.empty:
        available_channels = sorted(df[channel_col].unique())

        raise ValueError(
            f"No data found for channel '{channel}'. "
            f"Available channel examples: {available_channels[:10]}"
        )

    if condition is not None:
        conditions = [condition]
    else:
        conditions = sorted(channel_df[condition_col].unique())

    saved_paths = []

    for cond in conditions:

        sub = channel_df[channel_df[condition_col] == cond].copy()

        if sub.empty:
            print(f"Skipping channel={channel}, condition={cond}")
            continue

        sub = sub.sort_values(freq_col)

        freq = sub[freq_col].to_numpy(dtype=float)
        mean_psd = sub[mean_col].to_numpy(dtype=float)
        sem_psd = sub[sem_col].to_numpy(dtype=float)
        sem_yerr = clean_errorbar_values(sem_psd)

        selected_background = sub[
            "selected_background_1overf_db"
        ].to_numpy(dtype=float)

        alpha = sub["alpha_gaussian_db"].to_numpy(dtype=float)
        beta = sub["beta_gaussian_db"].to_numpy(dtype=float)
        best_model = sub["best_model_fit_db"].to_numpy(dtype=float)
        residual = sub["psd_db_bg_removed"].to_numpy(dtype=float)

        alpha_safe = np.where(np.isfinite(alpha), alpha, 0.0)
        beta_safe = np.where(np.isfinite(beta), beta, 0.0)

        selected_model = sub["selected_model"].iloc[0]
        selected_aic = sub["selected_model_aic"].iloc[0]

        alpha_mu = round(sub["alpha_mu_hz"].iloc[0], 2)
        alpha_var = round(sub["alpha_sigma_hz"].iloc[0]**2, 2)
        alpha_amp = round(sub["alpha_amp_db"].iloc[0], 1)

        beta_mu = round(sub["beta_mu_hz"].iloc[0], 2)
        beta_var = round(sub["beta_sigma_hz"].iloc[0]**2, 2)
        beta_amp = round(sub["beta_amp_db"].iloc[0], 1)


        peak_report = format_gaussian_peak_report(sub)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )

        ax_top, ax_bottom = axes

        ax_top.errorbar(
            freq,
            mean_psd,
            yerr=sem_yerr,
            fmt="-o",
            markersize=4,
            capsize=3,
            color='black',
            alpha=0.75,
            label="Mean PSD ± SEM",
        )

        ax_top.plot(
            freq,
            selected_background,
            "--",
            linewidth=2,
            color='orange',
            label="Background from Selected model",
        )

        ax_top.plot(
            freq,
            best_model,
            "-",
            color='green',
            linewidth=2,
            label="Selected model",
        )

        if np.isfinite(alpha).any():
            ax_top.plot(
                freq,
                selected_background + alpha_safe,
                ":",
                linewidth=2,
                color='red',
                label=r"Background + $\alpha$",
            )

        if np.isfinite(beta).any():
            if np.isfinite(alpha).any():
                beta_label = r"Background + $\alpha$ + $\beta$"
                beta_curve = selected_background + alpha_safe + beta_safe
            else:
                beta_label = r"Background + $\beta$"
                beta_curve = selected_background + beta_safe

            ax_top.plot(
                freq,
                beta_curve,
                ":",
                linewidth=2,
                color='blue',
                label=beta_label,
            )

        if np.isfinite(alpha_mu):
            ax_top.axvline(
                alpha_mu,
                linestyle=":",
                linewidth=1,
                color='red',
                label=f"Alpha peak ({alpha_amp}): N ~ ({alpha_mu:.2f}, {alpha_var})",
            )
            ax_bottom.axvline(
                alpha_mu,
                linestyle=":",
                linewidth=1,
                color='red',
            )

        if np.isfinite(beta_mu):
            ax_top.axvline(
                beta_mu,
                linestyle=":",
                linewidth=1,
                color='blue',
                label=f"Beta peak ({beta_amp}): N ~ ({beta_mu:.2f}, {beta_var})",
            )
            ax_bottom.axvline(
                beta_mu,
                linestyle=":",
                linewidth=1,
                color='blue',
            )

        ax_top.text(
            0.02,
            0.04,
            peak_report,
            transform=ax_top.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

        ax_top.set_ylabel("PSD (dB)")
        ax_top.set_title(
            f"Channel {channel} | Condition {cond} | "
            f"Best={selected_model} | AIC={selected_aic:.2f}"
        )
        ax_top.legend(loc="best")
        ax_top.grid(True, alpha=0.3)

        ax_top.axvline(8, lw=1, ls=':', color='grey', alpha=0.85)
        ax_top.axvline(13, lw=1, ls=':', color='grey', alpha=0.85)
        ax_top.axvline(30, lw=1, ls=':', color='grey', alpha=0.85)

        ax_bottom.axvline(8, lw=1, ls=':', color='grey', alpha=0.85)
        ax_bottom.axvline(13, lw=1, ls=':', color='grey', alpha=0.85)
        ax_bottom.axvline(30, lw=1, ls=':', color='grey', alpha=0.85)
        ax_bottom.axhline(0, linestyle="--", linewidth=1, color='grey', alpha=0.75)

        ax_bottom.errorbar(
            freq,
            residual,
            yerr=sem_yerr,
            fmt="-o",
            markersize=4,
            capsize=3,
            color='black',
            alpha=0.75,
            label="PSD after subtracting 1/f background ± SEM",
        )

        

        ax_bottom.text(
            0.02,
            0.04,
            "Residual error bars = SEM of mean PSD\n"
            "Model-fit uncertainty not propagated",
            transform=ax_bottom.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

        ax_bottom.set_xlabel("Frequency (Hz)")
        ax_bottom.set_ylabel("Residual PSD (dB)")
        ax_bottom.legend(loc="best")
        ax_bottom.grid(True, alpha=0.3)

        plt.tight_layout()

        out_path = (
            output_dir
            / f"psd_best_fit_channel_{safe_filename(channel)}"
              f"_condition_{safe_filename(cond)}.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(out_path)

    return saved_paths


def plot_best_fit_for_channels(
    df,
    channels,
    output_dir,
    condition=None,
    freq_col="frequency_hz",
    condition_col="condition",
    channel_col="channel",
    mean_col="mean_psd_db",
    sem_col="sem_psd_db",
):
    """
    Save PSD plots using the selected AIC-best model for multiple channels
    """

    if channels is None:
        channels = sorted(df[channel_col].dropna().unique())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for channel in channels:

        try:
            channel_paths = plot_best_fit_for_channel(
                df=df,
                channel=channel,
                output_dir=output_dir,
                condition=condition,
                freq_col=freq_col,
                condition_col=condition_col,
                channel_col=channel_col,
                mean_col=mean_col,
                sem_col=sem_col,
            )

            saved_paths.extend(channel_paths)

        except ValueError as exc:
            print(f"Skipping channel={channel}: {exc}")

    return saved_paths


# =============================================================================
# Full pipeline
# =============================================================================

def process_psd_file(
    input_csv,
    output_aggregated_csv="psd_long_format_aggregated.csv",
    output_best_fit_csv="psd_long_format_aggregated_best_fit.csv",
    output_fit_summary_csv="psd_fit_summary_all_models.csv",
    output_peak_summary_csv="psd_selected_gaussian_peak_summary.csv",
    output_peak_group_summary_csv="psd_selected_gaussian_peak_group_summary.csv",
    plot_channels=None,
    plot_condition=None,
    plot_folder_name="psd_best_fit_plots",
    fit_freq_min=1.0,
    fit_freq_max=40.0,
    alpha_mu_guess=10.0,
    beta_mu_guess=20.0,
    alpha_mu_bounds=(7.5, 13.0),
    beta_mu_bounds=(14.0, 30.0),
    alpha_sigma_bounds=(0.5, 2.5),
    beta_sigma_bounds=(0.5, 4.0),
    amp_bounds=(0.0, 30.0),
    use_sem_weights=True,
    min_peak_amp_db=0.5,
    boundary_tol_hz=0.05,
    min_delta_aic_for_extra_peak=3.0,
):
    """
    Run aggregation, model comparison, AIC selection, peak summary, and plotting
    """

    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = input_path.parent

    aggregated = aggregate_psd_db_over_batch(input_path)

    aggregated_output_path = output_dir / output_aggregated_csv
    aggregated.to_csv(aggregated_output_path, index=False)

    best_fit_df, fit_summary_df = add_best_model_fits_to_aggregated(
        aggregated_df=aggregated,
        fit_freq_min=fit_freq_min,
        fit_freq_max=fit_freq_max,
        alpha_mu_guess=alpha_mu_guess,
        beta_mu_guess=beta_mu_guess,
        alpha_mu_bounds=alpha_mu_bounds,
        beta_mu_bounds=beta_mu_bounds,
        alpha_sigma_bounds=alpha_sigma_bounds,
        beta_sigma_bounds=beta_sigma_bounds,
        amp_bounds=amp_bounds,
        use_sem_weights=use_sem_weights,
        min_peak_amp_db=min_peak_amp_db,
        boundary_tol_hz=boundary_tol_hz,
        min_delta_aic_for_extra_peak=min_delta_aic_for_extra_peak,
    )

    peak_summary_df = make_selected_gaussian_peak_summary(best_fit_df)
    peak_group_summary_df = summarize_selected_gaussian_peaks(peak_summary_df)

    best_fit_output_path = output_dir / output_best_fit_csv
    fit_summary_output_path = output_dir / output_fit_summary_csv
    peak_summary_output_path = output_dir / output_peak_summary_csv
    peak_group_summary_output_path = output_dir / output_peak_group_summary_csv

    best_fit_df.to_csv(best_fit_output_path, index=False)
    fit_summary_df.to_csv(fit_summary_output_path, index=False)
    peak_summary_df.to_csv(peak_summary_output_path, index=False)
    peak_group_summary_df.to_csv(peak_group_summary_output_path, index=False)

    plot_paths = []

    if plot_channels is not False:
        plot_output_dir = output_dir / plot_folder_name

        plot_paths = plot_best_fit_for_channels(
            df=best_fit_df,
            channels=plot_channels,
            condition=plot_condition,
            output_dir=plot_output_dir,
        )

    return {
        "aggregated_df": aggregated,
        "best_fit_df": best_fit_df,
        "fit_summary_df": fit_summary_df,
        "peak_summary_df": peak_summary_df,
        "peak_group_summary_df": peak_group_summary_df,
        "aggregated_csv": aggregated_output_path,
        "best_fit_csv": best_fit_output_path,
        "fit_summary_csv": fit_summary_output_path,
        "peak_summary_csv": peak_summary_output_path,
        "peak_group_summary_csv": peak_group_summary_output_path,
        "plot_paths": plot_paths,
    }


# =============================================================================
# Run script
# =============================================================================

if __name__ == "__main__":

    base_dir = Path("/mnt/c/Users/scana/Desktop/motorimagery_result_test")

    files = [
        #"sub-PDHC001_ses-01_task-MotorImag_run-01",
        #"sub-PDHC002_ses-01_task-MotorImag_run-01",
        #"sub-PDHC003_ses-01_task-MotorImag_run-01",
        #"sub-PDHC004_ses-01_task-MotorImag_run-01",
        #"sub-PDHC005_ses-01_task-MotorImag_run-01",
        #"sub-PDHC006_ses-01_task-MotorImag_run-01",
        #"sub-PDHC007_ses-01_task-MotorImag_run-01",
        #"sub-PDHC008_ses-01_task-MotorImag_run-01",
        #"sub-PDHC009_ses-01_task-MotorImag_run-01",
        #"sub-PDHC010_ses-01_task-MotorImag_run-01",
        #"sub-PDHC011_ses-01_task-MotorImag_run-01",
        #"sub-PDHC012_ses-01_task-MotorImag_run-01",
        #"sub-PDHC013_ses-01_task-MotorImag_run-01",
        #"sub-PDHC014_ses-01_task-MotorImag_run-01",
        #"sub-PDHC015_ses-01_task-MotorImag_run-01",
        #"sub-PDHC016_ses-01_task-MotorImag_run-01",
        #"sub-PDHC017_ses-01_task-MotorImag_run-01",
        #"sub-PDHC018_ses-01_task-MotorImag_run-01",
        #"sub-PDHC019_ses-01_task-MotorImag_run-01",
        #"sub-PDHC020_ses-01_task-MotorImag_run-01",
        #"sub-PDHC021_ses-01_task-MotorImag_run-01",
        #"sub-PDHC022_ses-01_task-MotorImag_run-01",
        #"sub-PDHC023_ses-01_task-MotorImag_run-01",
        #"sub-PDHC024_ses-01_task-MotorImag_run-01",
        #"sub-PDHC025_ses-01_task-MotorImag_run-01",
        #"sub-PDHC026_ses-01_task-MotorImag_run-01",
        #"sub-PDHC027_ses-01_task-MotorImag_run-01",
        #"sub-PDHC028_ses-01_task-MotorImag_run-01",
        #"sub-PDHC029_ses-01_task-MotorImag_run-01",
        #"sub-PDHC030_ses-01_task-MotorImag_run-01",
        #"sub-PDHC031_ses-01_task-MotorImag_run-01",
        #"sub-PDHC032_ses-01_task-MotorImag_run-01",
        #"sub-PDHC033_ses-01_task-MotorImag_run-01",
        #"sub-PDHC034_ses-01_task-MotorImag_run-01",
        #"sub-PDHC035_ses-01_task-MotorImag_run-01",
        #"sub-PDHC036_ses-01_task-MotorImag_run-01",
        "sub-PDHC038_ses-01_task-MotorImag_run-01",
        "sub-PDHC039_ses-01_task-MotorImag_run-01",
    ]

    input_filename = "psd_long_format.csv"

    # Use a list to plot specific channels
    #plot_channels = ["c1", "c3", "cp1", "c2", "c4", "cp2"]
    plot_channels = ["c3", "c4"]

    # Use None to plot all available channels
    # plot_channels = None

    # Use False to disable plotting
    # plot_channels = False

    plot_condition = None

    fit_freq_min = 1.0
    fit_freq_max = 40.0

    alpha_mu_guess = 10.0
    beta_mu_guess = 20.0

    alpha_mu_bounds = (7.5, 13.0)
    beta_mu_bounds = (14.0, 30.0)

    alpha_sigma_bounds = (0.5, 2.5)
    beta_sigma_bounds = (0.5, 4.0)

    amp_bounds = (0.0, 30.0)

    use_sem_weights = True

    min_peak_amp_db = 0.5
    boundary_tol_hz = 0.05
    min_delta_aic_for_extra_peak = 3.0

    for subject in files:

        input_csv = base_dir / subject / "csv" / input_filename

        print("\n" + "=" * 80)
        print(f"Subject: {subject}")
        print(f"Input:   {input_csv}")

        if not input_csv.exists():
            print(f"Skipping missing file: {input_csv}")
            continue

        results = process_psd_file(
            input_csv=input_csv,
            output_aggregated_csv="psd_long_format_aggregated.csv",
            output_best_fit_csv="psd_long_format_aggregated_best_fit.csv",
            output_fit_summary_csv="psd_fit_summary_all_models.csv",
            output_peak_summary_csv="psd_selected_gaussian_peak_summary.csv",
            output_peak_group_summary_csv="psd_selected_gaussian_peak_group_summary.csv",
            plot_channels=plot_channels,
            plot_condition=plot_condition,
            plot_folder_name="psd_best_fit_plots",
            fit_freq_min=fit_freq_min,
            fit_freq_max=fit_freq_max,
            alpha_mu_guess=alpha_mu_guess,
            beta_mu_guess=beta_mu_guess,
            alpha_mu_bounds=alpha_mu_bounds,
            beta_mu_bounds=beta_mu_bounds,
            alpha_sigma_bounds=alpha_sigma_bounds,
            beta_sigma_bounds=beta_sigma_bounds,
            amp_bounds=amp_bounds,
            use_sem_weights=use_sem_weights,
            min_peak_amp_db=min_peak_amp_db,
            boundary_tol_hz=boundary_tol_hz,
            min_delta_aic_for_extra_peak=min_delta_aic_for_extra_peak,
        )

        print(f"Saved aggregated CSV:            {results['aggregated_csv']}")
        print(f"Saved best-fit CSV:              {results['best_fit_csv']}")
        print(f"Saved fit-summary CSV:           {results['fit_summary_csv']}")
        print(f"Saved peak-summary CSV:          {results['peak_summary_csv']}")
        print(f"Saved peak-group-summary CSV:    {results['peak_group_summary_csv']}")

        selected_counts = (
            results["fit_summary_df"]
            .query("is_selected_best_model == True")
            ["model_name"]
            .value_counts()
        )

        print("\nSelected model counts:")
        print(selected_counts.to_string())

        for plot_path in results["plot_paths"]:
            print(f"Saved plot:                      {plot_path}")

    print("\nDone")