try:
    from _Cfilters import nullify_secondary_maxima
except ImportError:
    import numpy as np
    # Because of the way C imports work, nullify_secondary_maxima
    # is *called*, as in nullify_secondary_maxima().
    # For the pure Python variant, we do not want to call the function,
    # so we make nullify_secondary_maxima a wrapper than returns
    # the pure Python function that does the actual filtering.
    def _filter(a):
        target = a.size // 2
        target_val = a[target]
        if target_val == 0:
            return 0  # speedup trivial case
        if np.any(a[:target] > target_val):
            return 0
        if np.any(a[target + 1:] >= target_val):
            return 0
        return target_val
    def nullify_secondary_maxima():
        return _filter
