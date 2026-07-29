"""Data quality controls and outlier reporting."""
import pandas as pd


def clean_market_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicates, impute market gaps, and report IQR outliers.

    Values are not deleted for IQR outliers because extreme moves are often the
    crash signals this research aims to retain.
    """
    cleaned = frame.copy().sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    numeric = cleaned.select_dtypes(include="number").columns
    cleaned[numeric] = cleaned[numeric].ffill().bfill()
    cleaned = cleaned.dropna(subset=numeric, how="all")

    report_rows = []
    for column in numeric:
        values = cleaned[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report_rows.append({"feature": column, "lower_bound": lower, "upper_bound": upper, "outlier_count": int(((values < lower) | (values > upper)).sum())})
    return cleaned, pd.DataFrame(report_rows)
