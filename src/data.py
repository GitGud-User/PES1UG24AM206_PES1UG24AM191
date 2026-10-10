"""Data preparation with explicit catalog coverage and past-only features."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ['longitude', 'latitude', 'depth', 'magnitude', 'count_1d', 'count_7d', 'count_30d']
TEMPORAL_FEATURES = ['count_6h', 'time_since_prev_hours', 'mean_magnitude_24h', 'max_magnitude_24h']
ALL_FEATURES = FEATURES + TEMPORAL_FEATURES
ROOT = Path(__file__).resolve().parents[1]


def past_counts(times, days):
    """Count events in [t-days, t), excluding every event at the current timestamp."""
    t = pd.DatetimeIndex(times).to_numpy(dtype='datetime64[ns]')
    if np.any(t[1:] < t[:-1]):
        raise ValueError('Earthquakes must be sorted by time')
    return np.searchsorted(t, t, side='left') - np.searchsorted(t, t - np.timedelta64(days, 'D'), side='left')


def past_temporal_features(times, magnitudes):
    """Build strictly past-only short-window earthquake features.

    ``searchsorted(..., side='left')`` excludes every event at the current
    timestamp, including duplicate timestamps. Empty windows use zero.
    """
    t = pd.DatetimeIndex(times).to_numpy(dtype='datetime64[ns]')
    m = np.asarray(magnitudes, dtype=float)
    if len(t) != len(m) or np.any(t[1:] < t[:-1]):
        raise ValueError('Times and magnitudes must be aligned and sorted')
    left_6h = np.searchsorted(t, t - np.timedelta64(6, 'h'), side='left')
    end = np.searchsorted(t, t, side='left')
    count_6h = end - left_6h
    left_24h = np.searchsorted(t, t - np.timedelta64(24, 'h'), side='left')
    sums = np.concatenate(([0.0], np.cumsum(m)))
    means = np.zeros(len(t))
    maxima = np.zeros(len(t))
    for i, (start, stop) in enumerate(zip(left_24h, end)):
        if stop > start:
            window = m[start:stop]
            means[i] = (sums[stop] - sums[start]) / (stop - start)
            maxima[i] = window.max()
    previous = end - 1
    since_hours = np.zeros(len(t))
    valid = previous >= 0
    since_hours[valid] = (t[valid] - t[previous[valid]]) / np.timedelta64(1, 'h')
    return np.column_stack((count_6h, since_hours, means, maxima))


def assign_labels(times, starts, lengths_hours, buffer_hours=24):
    """Date-only starts are midnight; [start, end+buffer) defines erupting.

    Unknown time to the next eruption is NaN, never an artificial zero.
    Latest recorded start determines the episode, matching the source convention.
    """
    t = pd.DatetimeIndex(times).to_numpy(dtype='datetime64[ns]')
    starts = pd.DatetimeIndex(starts).to_numpy(dtype='datetime64[ns]')
    if len(starts) == 0 or np.any(starts[1:] <= starts[:-1]):
        raise ValueError('Eruption starts must be nonempty and strictly increasing')
    durations = np.asarray(lengths_hours, float)
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError('Invalid eruption durations')
    ends = starts + ((durations + buffer_hours) * 3600e9).astype('timedelta64[ns]')
    previous = np.searchsorted(starts, t, side='right') - 1
    active = (previous >= 0) & (t < ends[np.maximum(previous, 0)])
    next_idx = np.searchsorted(starts, t, side='right')
    target = np.full(len(t), np.nan)
    known = next_idx < len(starts)
    target[known] = (starts[next_idx[known]] - t[known]) / np.timedelta64(1, 'D')
    target[active] = 0.0
    available = t.copy()
    available[known & ~active] = starts[next_idx[known & ~active]]
    return active.astype(int), target, available


def prepare(raw_dir=None, buffer_hours=24):
    raw_dir = Path(raw_dir or ROOT / 'data/raw')
    eq = pd.read_csv(raw_dir / 'puuoo_earthquakes.csv')
    eruptions = pd.read_csv(raw_dir / 'PuuOo.csv')
    audit = {'raw_earthquakes': len(eq), 'raw_eruptions': len(eruptions), 'buffer_hours': buffer_hours}
    audit['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.glob('*.csv')}
    eq['source_row'] = np.arange(len(eq))
    eq = eq.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'Depth': 'depth', 'Magnitude': 'magnitude'})
    eq['time'] = pd.to_datetime(eq['Date-time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    for col in FEATURES[:4]:
        eq[col] = pd.to_numeric(eq[col], errors='coerce')
        eq.loc[eq[col] == -9999, col] = np.nan
    valid = eq['time'].notna() & np.isfinite(eq[FEATURES[:4]]).all(axis=1)
    audit['invalid_rows_removed'] = int((~valid).sum())
    eq = eq[valid].drop_duplicates(subset=['time'] + FEATURES[:4]).sort_values('time', kind='stable')
    audit['duplicates_removed'] = int(valid.sum() - len(eq))
    # Same geographic half-plane as GetEastRiftZoneEQs in RunMLTests.ipynb.
    eq = eq[eq.latitude <= 0.4 * eq.longitude + 81.5].copy()
    audit['east_rift_earthquakes'] = len(eq)
    for days in (1, 7, 30):
        eq[f'count_{days}d'] = past_counts(eq.time, days)
    eq[TEMPORAL_FEATURES] = past_temporal_features(eq.time, eq.magnitude)
    starts = pd.to_datetime(eruptions['Date'], format='%m/%d/%y')
    hours = eruptions['Length'].str.extract(r'^([\d.]+) hrs$')[0].astype(float)
    # The source catalog has one out-of-order pair (rows 35.5 and 35).
    # Keep onset/duration pairs together and process episodes chronologically.
    eruption_order = np.argsort(starts.to_numpy(), kind='stable')
    starts = starts.iloc[eruption_order].reset_index(drop=True)
    hours = hours.iloc[eruption_order].reset_index(drop=True)
    # Episode 48 says 173566800 hours; do not guess a corrected duration.
    # Its known onset may label earlier repose events, but exclude data at/after it.
    suspect = hours > 24 * 366
    if list(eruptions.loc[suspect, 'Number']) != [48]:
        raise ValueError('Unexpected eruption-duration anomaly: review the source data')
    cutoff = starts[suspect].min()
    safe_hours = hours.mask(suspect, 0)
    eq['erupting'], eq['tte_days'], eq['label_available_time'] = assign_labels(eq.time, starts, safe_hours, buffer_hours)
    warmup_end = eq.time.min() + pd.Timedelta(days=30)
    keep = (eq.time >= max(starts.min(), warmup_end)) & (eq.time < cutoff) & eq.tte_days.notna()
    audit.update({'coverage_cutoff_exclusive': str(cutoff), 'warmup_end': str(warmup_end),
                  'coverage_or_warmup_rows_removed': int((~keep).sum())})
    data = eq.loc[keep, ['source_row','time'] + ALL_FEATURES + ['erupting','tte_days','label_available_time']].reset_index(drop=True)
    audit.update({'model_rows': len(data), 'positive_rows': int(data.erupting.sum()),
                  'positive_fraction': float(data.erupting.mean()), 'start': str(data.time.min()), 'end': str(data.time.max())})
    return data, audit
