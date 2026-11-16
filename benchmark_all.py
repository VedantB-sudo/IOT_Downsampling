import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import METHODS, choose_samples
from signal_quality import evaluate_fidelity

DATA = Path('data/PAMAP2_Dataset')
OUTPUT = Path('benchmark_results/pamap2/full_dataset')
CHANNELS = {
    'hand_acc_x': 4, 'hand_acc_y': 5, 'hand_acc_z': 6,
    'hand_gyro_x': 10, 'hand_gyro_y': 11, 'hand_gyro_z': 12,
    'hand_mag_x': 13, 'hand_mag_y': 14, 'hand_mag_z': 15,
    'chest_acc_x': 21, 'chest_acc_y': 22, 'chest_acc_z': 23,
    'ankle_acc_x': 38, 'ankle_acc_y': 39, 'ankle_acc_z': 40
}
DEFAULT_CHANNELS = ['hand_acc_x', 'hand_gyro_x', 'hand_mag_x']
REDUCTIONS = [50, 60, 70, 80]
COLUMNS = [
    'recording', 'subject', 'activity_id', 'sensor_type', 'source_start_row',
    'original_samples', 'algorithm', 'target_reduction_pct',
    'data_reduction_pct', 'retained_samples', 'RMSE', 'MAE', 'PRDN_pct',
    'P95_retained_pct', 'P99_retained_pct', 'selection_time_ms'
]
KEYS = ['recording', 'subject', 'activity_id', 'sensor_type',
        'source_start_row', 'original_samples', 'algorithm',
        'target_reduction_pct']


def load_recording(path, channels):
    columns = sorted({0, 1, *(CHANNELS[name] for name in channels)})
    return pd.read_csv(path, sep=r'\s+', header=None, usecols=columns)


def valid_windows(df, channels, size, minimum):
    timestamps = df[0].to_numpy()
    activities = df[1].to_numpy()
    valid = (activities > 0) & np.isfinite(timestamps)

    for channel in channels:
        valid &= np.isfinite(df[CHANNELS[channel]].to_numpy())

    boundaries = np.flatnonzero(
        ~valid[1:] | ~valid[:-1]
        | (activities[1:] != activities[:-1])
        | (np.diff(timestamps) <= 0)
        | (np.diff(timestamps) > 0.025)
    ) + 1

    for left, right in zip(np.r_[0, boundaries], np.r_[boundaries, len(df)]):
        if not valid[left]:
            continue
        for start in range(int(left), int(right), size):
            end = min(start + size, int(right))
            if end - start >= minimum:
                yield int(activities[start]), start, end


def row_key(row):
    return tuple(str(row[name]) for name in KEYS)


def summarise(csv_path, output):
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    metrics = ['PRDN_pct', 'RMSE', 'MAE', 'P95_retained_pct',
               'P99_retained_pct', 'selection_time_ms']
    for label, keys in [
        ('benchmark_summary.csv', ['sensor_type', 'target_reduction_pct', 'algorithm']),
        ('benchmark_by_activity.csv', ['recording', 'activity_id', 'sensor_type',
                                       'target_reduction_pct', 'algorithm'])
    ]:
        summary = df.groupby(keys, dropna=False)[metrics].agg(['mean', 'std'])
        summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
        summary['window_count'] = df.groupby(keys).size()
        summary.reset_index().to_csv(output / label, index=False)


def main():
    parser = argparse.ArgumentParser(description='Full PAMAP2 offline benchmark')
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(101, 110)))
    parser.add_argument('--channels', nargs='+', choices=list(CHANNELS), default=DEFAULT_CHANNELS)
    parser.add_argument('--protocol-only', action='store_true')
    parser.add_argument('--window-samples', type=int, default=1000)
    parser.add_argument('--min-samples', type=int, default=256)
    parser.add_argument('--reductions', nargs='+', type=int, default=REDUCTIONS)
    parser.add_argument('--window', type=int, default=48)
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--hp-alpha', type=float, default=0.1)
    parser.add_argument('--pe-weight', type=float, default=0.5)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-windows', type=int, default=0,
                        help='Limit windows for a test run (0 means all)')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not 100 <= args.min_samples <= args.window_samples:
        parser.error('Require 100 <= min-samples <= window-samples')
    if args.window < 6 or args.chunk < 8 or args.repeats < 1:
        parser.error('Invalid window, chunk, or repeats')
    if any(not 0 < reduction < 100 for reduction in args.reductions):
        parser.error('Reductions must be between 1 and 99')
    if not 0 < args.hp_alpha <= 1 or not 0 <= args.pe_weight <= 1:
        parser.error('Invalid high-pass alpha or PE weight')
    if args.max_windows < 0 or any(s < 101 or s > 109 for s in args.subjects):
        parser.error('Invalid max-windows or subjects')
    if len(set(args.reductions)) != len(args.reductions):
        parser.error('Duplicate reduction levels are not allowed')
    if len(set(args.channels)) != len(args.channels) or len(set(args.subjects)) != len(args.subjects):
        parser.error('Duplicate channels or subjects are not allowed')

    configuration = {
        'subjects': args.subjects, 'channels': args.channels,
        'protocol_only': args.protocol_only, 'window_samples': args.window_samples,
        'min_samples': args.min_samples, 'reductions': args.reductions,
        'window': args.window, 'chunk': args.chunk,
        'hp_alpha': args.hp_alpha, 'pe_weight': args.pe_weight,
        'repeats': args.repeats, 'methods': METHODS,
        'reconstruction': 'Linear interpolation by sample index',
        'event_definition': 'P95/P99 exact-index high-change retention',
        'network_tested': False
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'benchmark_settings.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        parser.error('Existing output uses different settings. Choose another --output folder.')
    config_path.write_text(json.dumps(configuration, indent=2), encoding='utf-8')

    csv_path = args.output / 'benchmark_trials.csv'
    completed = set()
    if csv_path.exists():
        old = pd.read_csv(csv_path, usecols=KEYS, dtype=str)
        completed = {row_key(row) for row in old.to_dict('records')}
        print(f'Resuming: {len(completed)} completed comparisons found')

    locations = ['Protocol'] if args.protocol_only else ['Protocol', 'Optional']
    files = [DATA / location / f'subject{subject}.dat'
             for location in locations for subject in args.subjects]
    files = [path for path in files if path.is_file()]
    if not files:
        parser.error(f'No PAMAP2 recordings found inside {DATA}')

    windows = 0
    new_rows = 0
    with csv_path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if csv_path.stat().st_size == 0:
            writer.writeheader()

        for path in files:
            df = load_recording(path, args.channels)
            subject = int(path.stem.replace('subject', ''))
            count = 0

            for activity, start, end in valid_windows(
                df, args.channels, args.window_samples, args.min_samples
            ):
                windows += 1
                count += 1

                for channel in args.channels:
                    signal = df[CHANNELS[channel]].to_numpy()[start:end]
                    for reduction in args.reductions:
                        keep = (100 - reduction) / 100
                        expected = max(2, min(len(signal), round(len(signal) * keep)))

                        for method in METHODS:
                            identity = {
                                'recording': path.parent.name, 'subject': subject,
                                'activity_id': activity, 'sensor_type': channel,
                                'source_start_row': start, 'original_samples': len(signal),
                                'algorithm': method, 'target_reduction_pct': reduction
                            }
                            if row_key(identity) in completed:
                                continue

                            timings = []
                            for _ in range(args.repeats):
                                begin = time.perf_counter()
                                indices = np.asarray(choose_samples(
                                    signal, method, keep, args.window,
                                    args.chunk, args.hp_alpha, args.pe_weight
                                ))
                                timings.append((time.perf_counter() - begin) * 1000)

                            if (indices.ndim != 1 or
                                not np.issubdtype(indices.dtype, np.integer) or
                                len(indices) != expected or
                                len(np.unique(indices)) != expected or
                                np.any(indices < 0) or np.any(indices >= len(signal))):
                                raise ValueError(f'Invalid indices: {path.name}, {method}')

                            metrics = evaluate_fidelity(signal, indices)
                            writer.writerow({
                                **identity,
                                'data_reduction_pct': 100 * (1 - len(indices) / len(signal)),
                                'retained_samples': len(indices), **metrics,
                                'selection_time_ms': float(np.median(timings))
                            })
                            new_rows += 1
                handle.flush()
                if args.max_windows and windows >= args.max_windows:
                    break

            print(f'{path.parent.name}/{path.name}: {count} windows')
            if args.max_windows and windows >= args.max_windows:
                break

    summarise(csv_path, args.output)
    print(f'\nProcessed {windows} windows; added {new_rows} comparisons')
    print(f'Trial results: {csv_path}')
    print(f'Summary: {args.output / "benchmark_summary.csv"}')
    print(f'By activity: {args.output / "benchmark_by_activity.csv"}')


if __name__ == '__main__':
    main()