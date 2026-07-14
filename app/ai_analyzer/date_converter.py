import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def is_excel_serial_date(value) -> bool:
    """Excel serial dates must be large AND not business numerics like hours, percent, scores, ratings, age."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    if pd.isna(value):
        return False

    # Immediately reject all percentage, age, hours, scores, ratings
    if 0 <= value <= 10000:
        return False
    if 1 <= value <= 20 and not float(value).is_integer():
        return False

    v = round(value)

    # Strict modern Excel serial range
    if v < 30000 or v > 60000:
        return False

    # Reject if decimal part is too noisy
    if abs(value - v) > 0.25:
        return False

    return True


def is_string_date(value):
    """Check for common date string formats."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    patterns = [
        r'^\d{1,2}/\d{1,2}/\d{2,4}$',
        r'^\d{1,2}-\d{1,2}-\d{2,4}$',
        r'^\d{4}-\d{1,2}-\d{1,2}$'
    ]
    for pat in patterns:
        if pd.Series(value).str.match(pat).any():
            return True
    try:
        d = datetime.fromisoformat(value.replace("/", "-"))
        if 1990 < d.year < 2050:
            return True
    except:
        return False
    return False


def excel_serial_to_date(excel_serial_date) -> datetime:
    """Convert Excel serial to datetime."""
    excel_epoch = datetime(1899, 12, 30)
    if isinstance(excel_serial_date, float):
        days = int(excel_serial_date)
        seconds = int((excel_serial_date - days) * 86400)
        return excel_epoch + timedelta(days=days, seconds=seconds)
    return excel_epoch + timedelta(days=int(excel_serial_date))


def detect_date_columns(df: pd.DataFrame) -> list:
    """Detect only valid Excel serial or string date columns and ignore percent/score/hours/age/id columns completely."""
    date_columns = []
    # Expanded block words to match frontend and avoid false positives
    block_words = ['percent', 'score', 'hours', 'rating', 'age', 'id', 'marks', 'qty', 'quantity', 'amount', 'price', 'index', 'no.', 'num']

    for col in df.columns:
        col_lower = str(col).lower()

        # ❌ If header contains block words, NEVER mark as date
        if any(b in col_lower for b in block_words):
            continue

        sample = df[col].dropna().head(20)
        if sample.empty:
            continue

        # Detect string dates first
        if sample.apply(lambda x: is_string_date(x)).sum() > len(sample) * 0.6:
            date_columns.append(col)
            continue

        # Detect excel serial only if numeric and passes strict rule
        if pd.api.types.is_numeric_dtype(sample):
            valid_serials = [v for v in sample if is_excel_serial_date(v)]
            
            # If mostly serial dates
            if len(valid_serials) > len(sample) * 0.7:
                median_val = np.median(valid_serials)
                # Check range (approx 1982 - 2064)
                if 30000 < median_val < 60000:
                    # Extra confidence if header looks like a date
                    if any(w in col_lower for w in ['date', 'time', 'dob', 'day', 'start', 'end', 'created', 'updated']):
                        date_columns.append(col)
                    else:
                        # If header is ambiguous but data strongly looks like dates (and not blocked), accept it
                        date_columns.append(col)

    return date_columns


def convert_excel_dates_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert detected date columns safely."""
    df_copy = df.copy()
    date_columns = detect_date_columns(df_copy)

    if not date_columns:
            return df_copy
        
    def format_date_to_string(dt: datetime) -> str:
            """Format datetime object to string in YYYY-MM-DD format."""
            if isinstance(dt, datetime):
                return dt.strftime("%Y-%m-%d")
            return str(dt)

    print(f"[DATE_CONVERTER] Safe detected {len(date_columns)} date columns: {date_columns}")

    for col in date_columns:
        try:
            df_copy[col] = df_copy[col].apply(
                lambda x: format_date_to_string(excel_serial_to_date(x)) if is_excel_serial_date(x)
                else format_date_to_string(datetime.fromisoformat(str(x).replace("/", "-"))) if is_string_date(x)
                else x
            )
        except:
            pass

    return df_copy
