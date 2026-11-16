from . import uniform_decimation, standard_deviation, hp_pe, hp_energy, hybrid, spaced_hybrid


ALGORITHMS = {
    "uniform": uniform_decimation.select_samples,
    "std_dev": standard_deviation.select_samples,
    "hp_pe": hp_pe.select_samples,
    "hp_energy": hp_energy.select_samples,
    "hybrid": hybrid.select_samples,
    "spaced_hybrid": spaced_hybrid.select_samples
}

METHODS = list(ALGORITHMS)


def choose_samples(x, method, keep, window=48, chunk=256, alpha=0.1, pe_weight=0.5):
    if method not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {method}")

    if method == "uniform":
        return ALGORITHMS[method](x, keep)
    if method == "std_dev":
        return ALGORITHMS[method](x, keep, window, chunk)

    if method in ("hybrid", "spaced_hybrid"):
        return ALGORITHMS[method](x, keep, window, chunk, alpha, pe_weight)

    return ALGORITHMS[method](x, keep, window, chunk, alpha)